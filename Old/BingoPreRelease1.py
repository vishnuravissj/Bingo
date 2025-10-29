# app.py - complete Tambola (Housie) app with admin key + claims + 4-corners
import streamlit as st
import random
import io
import zipfile
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from typing import Dict, List

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="Tambola - Full App", layout="wide")
ADMIN_KEY = "TAMBOLA2025"   # Change this before sharing
DEFAULT_TICKET_COUNT = 200
TICKET_ROWS, TICKET_COLS = 3, 9
TICKET_IMG_W, TICKET_IMG_H = 540, 300
FONT_PATH = None  # Optional path to .ttf if you want nicer fonts

# -------------------------
# SESSION STATE INIT (Pylance-safe)
# -------------------------
st.session_state.setdefault("tickets", {})               # Dict[str, List[List[int]]]
st.session_state.setdefault("called_numbers", [])        # List[int]
st.session_state.setdefault("claimed_prizes", {})        # Dict[str, str] prize -> ticket_id
st.session_state.setdefault("claims_queue", [])          # List[dict] newest first
st.session_state.setdefault("game_started", False)
st.session_state.setdefault("last_draw", None)
st.session_state.setdefault("prize_config", {
    "Early 5": True, "Top Line": True, "Middle Line": True,
    "Bottom Line": True, "4 Corners": True, "Full House": True
})
st.session_state.setdefault("loaded_ticket_id", None)    # persistence after Load
st.session_state.setdefault("admin_unlocked", False)

# -------------------------
# UTILS: Ticket generation (robust)
# -------------------------
COLUMN_RANGES = [(1,10),(11,20),(21,30),(31,40),(41,50),(51,60),(61,70),(71,80),(81,90)]

def generate_single_ticket():
    """Return one valid 3x9 ticket where each row has 5 numbers and columns follow ranges."""
    while True:
        # choose counts per column: total 15, each <=3
        counts = [0]*9
        rem = 15
        while rem > 0:
            i = random.randrange(9)
            if counts[i] < 3:
                counts[i] += 1
                rem -= 1
        cols = []
        valid = True
        for i, cnt in enumerate(counts):
            lo, hi = COLUMN_RANGES[i]
            pool = list(range(lo, hi+1))
            if cnt > len(pool):
                valid = False
                break
            cols.append(sorted(random.sample(pool, cnt)) if cnt > 0 else [])
        if not valid:
            continue
        # distribute into 3 rows ensuring each row gets 5 numbers
        grid = [[None]*9 for _ in range(3)]
        ok = True
        for col_idx, nums in enumerate(cols):
            rows = list(range(3))
            random.shuffle(rows)
            for n in nums:
                placed = False
                # pick the row with least numbers currently (stable)
                rows_sorted = sorted(rows, key=lambda r: sum(1 for x in grid[r] if x is not None))
                for r in rows_sorted:
                    if grid[r][col_idx] is None and sum(1 for x in grid[r] if x is not None) < 5:
                        grid[r][col_idx] = n
                        placed = True
                        break
                if not placed:
                    # fallback scan
                    for r in range(3):
                        if grid[r][col_idx] is None and sum(1 for x in grid[r] if x is not None) < 5:
                            grid[r][col_idx] = n
                            placed = True
                            break
                if not placed:
                    ok = False
                    break
            if not ok:
                break
        if ok and all(sum(1 for x in grid[r] if x is not None) == 5 for r in range(3)):
            # replace None with 0 (for consistency)
            final = [[(cell if cell is not None else 0) for cell in row] for row in grid]
            return final
        # else retry

def gen_unique_tickets(n: int, prefix: str = "T") -> Dict[str, List[List[int]]]:
    tickets = {}
    seen = set()
    attempts = 0
    while len(tickets) < n:
        attempts += 1
        if attempts > n * 2000:
            raise RuntimeError("Too many attempts generating unique tickets.")
        grid = generate_single_ticket()
        key = tuple(sorted(x for row in grid for x in row if x != 0))
        if key in seen:
            continue
        seen.add(key)
        tid = f"{prefix}{len(tickets)+1:04d}"
        tickets[tid] = grid
    return tickets

# -------------------------
# RENDERING & EXPORTS
# -------------------------
def render_ticket_image(grid, ticket_id):
    img = Image.new("RGB", (TICKET_IMG_W, TICKET_IMG_H), color=(255,255,255))
    draw = ImageDraw.Draw(img)
    try:
        if FONT_PATH:
            header_font = ImageFont.truetype(FONT_PATH, 18)
            cell_font = ImageFont.truetype(FONT_PATH, 16)
        else:
            header_font = ImageFont.load_default()
            cell_font = ImageFont.load_default()
    except Exception:
        header_font = ImageFont.load_default()
        cell_font = ImageFont.load_default()

    draw.text((10,6), f"Ticket {ticket_id}", font=header_font, fill=(0,0,0))
    left, top = 10, 40
    cell_w = (TICKET_IMG_W - left*2) // TICKET_COLS
    cell_h = (TICKET_IMG_H - top - 10) // TICKET_ROWS

    for r in range(TICKET_ROWS):
        for c in range(TICKET_COLS):
            x0 = left + c*cell_w
            y0 = top + r*cell_h
            x1, y1 = x0 + cell_w, y0 + cell_h
            draw.rectangle([x0,y0,x1,y1], outline=(0,0,0), width=1)
            val = grid[r][c]
            if val and val != 0:
                txt = str(val)
                try:
                    bbox = draw.textbbox((0,0), txt, font=cell_font)
                    w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
                except AttributeError:
                    w, h = draw.textsize(txt, font=cell_font)
                draw.text((x0 + (cell_w-w)/2, y0 + (cell_h-h)/2), txt, font=cell_font, fill=(0,0,0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

def tickets_to_dataframe(tickets: Dict[str, List[List[int]]]) -> pd.DataFrame:
    rows = []
    for tid, grid in tickets.items():
        nums = [str(n) for row in grid for n in row if n != 0]
        rows.append({"ticket_id": tid, "numbers": ",".join(nums)})
    return pd.DataFrame(rows)

def create_tickets_zip(tickets: Dict[str, List[List[int]]]):
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w") as zf:
        for tid, grid in tickets.items():
            b = render_ticket_image(grid, tid)
            zf.writestr(f"{tid}.png", b.read())
    mem.seek(0)
    return mem

# -------------------------
# PRIZE CHECKS
# -------------------------
def ticket_number_set(grid):
    return set(n for row in grid for n in row if n != 0)

def check_prize_for_ticket(grid, called_set, prize_name):
    # called_set is a set of ints
    if prize_name == "Early 5":
        return len(ticket_number_set(grid) & called_set) >= 5
    if prize_name == "Top Line":
        top = set(n for n in grid[0] if n != 0)
        return top.issubset(called_set)
    if prize_name == "Middle Line":
        mid = set(n for n in grid[1] if n != 0)
        return mid.issubset(called_set)
    if prize_name == "Bottom Line":
        bot = set(n for n in grid[2] if n != 0)
        return bot.issubset(called_set)
    if prize_name == "4 Corners":
        corners = [grid[0][0], grid[0][8], grid[2][0], grid[2][8]]
        return all((c != 0 and c in called_set) for c in corners)
    if prize_name == "Full House":
        return ticket_number_set(grid).issubset(called_set)
    return False

# -------------------------
# CLAIMS
# -------------------------
def submit_claim(ticket_id: str, claimant: str, claim_type: str) -> int:
    cid = len(st.session_state.claims_queue) + 1
    rec = {
        "id": cid,
        "ticket_id": ticket_id,
        "claimant": claimant,
        "claim_type": claim_type,
        "time": datetime.utcnow().isoformat(),
        "status": "pending",
        "validated_by": None,
        "validated_at": None
    }
    st.session_state.claims_queue.insert(0, rec)
    return cid

def validate_claim(cid: int, approver: str, approve: bool):
    for rec in st.session_state.claims_queue:
        if rec["id"] == cid:
            rec["status"] = "approved" if approve else "rejected"
            rec["validated_by"] = approver
            rec["validated_at"] = datetime.utcnow().isoformat()
            if approve and st.session_state.prize_config.get(rec["claim_type"], False):
                if rec["claim_type"] not in st.session_state.claimed_prizes:
                    st.session_state.claimed_prizes[rec["claim_type"]] = rec["ticket_id"]
            return True
    return False

# -------------------------
# ADMIN SIDEBAR + AUTH
# -------------------------
with st.sidebar:
    st.header("Admin Panel")
    key = st.text_input("Enter Admin Key", type="password")
    if st.button("Unlock Admin"):
        if key == ADMIN_KEY:
            st.session_state.admin_unlocked = True
            st.success("Admin unlocked")
        else:
            st.error("Invalid key")

# -------------------------
# ADMIN CONTROLS (secure)
# -------------------------
if st.session_state.admin_unlocked:
    st.sidebar.markdown("---")
    st.sidebar.header("Server Controls")

    # prizes - dynamic
    st.sidebar.subheader("Active Prizes")
    # ensure 4 Corners present in config
    if "4 Corners" not in st.session_state.prize_config:
        st.session_state.prize_config["4 Corners"] = True
    for pname in list(st.session_state.prize_config.keys()):
        st.session_state.prize_config[pname] = st.sidebar.checkbox(pname, value=st.session_state.prize_config[pname])

    st.sidebar.markdown("---")
    st.sidebar.subheader("Tickets")
    gen_n = st.sidebar.number_input("Generate how many tickets (overwrite)", min_value=1, max_value=2000, value=DEFAULT_TICKET_COUNT)
    prefix = st.sidebar.text_input("Ticket prefix", value="T")
    if st.sidebar.button("Generate Tickets (overwrite)"):
        st.session_state.tickets = gen_unique_tickets(gen_n, prefix)
        st.session_state.called_numbers = []
        st.session_state.claimed_prizes = {}
        st.session_state.claims_queue = []
        st.sidebar.success(f"Generated {len(st.session_state.tickets)} tickets")

    if st.sidebar.button("Export tickets CSV"):
        if not st.session_state.tickets:
            st.sidebar.warning("No tickets to export")
        else:
            df = tickets_to_dataframe(st.session_state.tickets)
            csv = df.to_csv(index=False).encode()
            st.sidebar.download_button("Download CSV", csv, "tickets.csv", "text/csv")

    if st.sidebar.button("Export ticket PNGs (ZIP)"):
        if not st.session_state.tickets:
            st.sidebar.warning("No tickets to export")
        else:
            z = create_tickets_zip(st.session_state.tickets)
            st.sidebar.download_button("Download ZIP", z, "tickets_images.zip", "application/zip")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Number Caller")
    if st.sidebar.button("Call Random Number"):
        remaining = set(range(1,91)) - set(st.session_state.called_numbers)
        if not remaining:
            st.sidebar.warning("All numbers called")
        else:
            n = random.choice(list(remaining))
            st.session_state.called_numbers.append(n)
            st.session_state.last_draw = n
            st.sidebar.success(f"Called {n}")

    manual = st.sidebar.number_input("Manual call (1-90)", min_value=1, max_value=90, value=1)
    if st.sidebar.button("Call Manual Number"):
        if manual in st.session_state.called_numbers:
            st.sidebar.warning(f"{manual} already called")
        else:
            st.session_state.called_numbers.append(manual)
            st.session_state.last_draw = manual
            st.sidebar.success(f"Called {manual}")

    if st.sidebar.button("Reset called numbers (keep tickets)"):
        st.session_state.called_numbers = []
        st.sidebar.success("Called numbers reset")

# -------------------------
# MAIN UI
# -------------------------
st.title("🎯 Tambola / Housie — Full App")
col1, col2 = st.columns([2,1])
with col2:
    st.metric("Numbers Called", f"{len(st.session_state.called_numbers)}/90")
    if st.session_state.last_draw:
        st.info(f"Last draw: {st.session_state.last_draw}")

st.header("Numbers Board")
called_sorted = sorted(st.session_state.called_numbers)
if called_sorted:
    grid_cols = st.columns(9)
    for i, n in enumerate(called_sorted):
        with grid_cols[i % 9]:
            st.button(str(n), key=f"num_{n}")
else:
    st.info("No numbers called yet")

# Auto-scan to mark prize as eligible (does not auto-award, only marks that eligible exists)
called_set = set(st.session_state.called_numbers)
for prize, enabled in st.session_state.prize_config.items():
    if not enabled:
        continue
    if prize in st.session_state.claimed_prizes:
        continue
    for tid, grid in st.session_state.tickets.items():
        if check_prize_for_ticket(grid, called_set, prize):
            st.session_state.claimed_prizes.setdefault(prize, None)
            break

# -------------------------
# PLAYER: load ticket (persist) + submit claim
# -------------------------
st.header("Player Portal — Load Ticket & Claim")
pid = st.text_input("Ticket ID", value=st.session_state.get("loaded_ticket_id") or "")

if st.button("Load Ticket"):
    if not pid:
        st.warning("Enter Ticket ID")
    else:
        if pid not in st.session_state.tickets:
            # create one if not present
            st.session_state.tickets[pid] = generate_single_ticket()
        st.session_state.loaded_ticket_id = pid
        st.success(f"Loaded ticket {pid}")

if st.session_state.get("loaded_ticket_id"):
    tid = st.session_state.loaded_ticket_id
    if tid in st.session_state.tickets:
        grid = st.session_state.tickets[tid]
        # render image and highlight called numbers
        img_b = render_ticket_image(grid, tid)
        img = Image.open(img_b)
        draw = ImageDraw.Draw(img)
        try:
            cell_font = ImageFont.truetype(FONT_PATH, 14) if FONT_PATH else ImageFont.load_default()
        except Exception:
            cell_font = ImageFont.load_default()
        left, top = 10, 40
        cw = (TICKET_IMG_W - left*2) // TICKET_COLS
        ch = (TICKET_IMG_H - top - 10) // TICKET_ROWS
        called = set(st.session_state.called_numbers)
        for r in range(TICKET_ROWS):
            for c in range(TICKET_COLS):
                val = grid[r][c]
                if val not in (None, 0) and val in called:
                    x0 = left + c*cw + 2
                    y0 = top + r*ch + 2
                    x1 = x0 + cw - 4
                    y1 = y0 + ch - 4
                    draw.rectangle([x0,y0,x1,y1], fill=(200,255,200))
                    txt = str(val)
                    try:
                        bbox = draw.textbbox((0,0), txt, font=cell_font)
                        w = bbox[2]-bbox[0]; h = bbox[3]-bbox[1]
                    except AttributeError:
                        w,h = draw.textsize(txt, font=cell_font)
                    draw.text((x0 + (cw-4-w)/2, y0 + (ch-4-h)/2), txt, font=cell_font, fill=(0,0,0))
        buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
        st.image(buf)

        st.write("Numbers called so far:", sorted(list(called)))
        st.markdown("### Submit Claim")
        claimant_name = st.text_input("Your name", key=f"name_{tid}")
        active_prizes = [p for p,en in st.session_state.prize_config.items() if en]
        claim_type = st.selectbox("Claim type", active_prizes, key=f"claim_{tid}")
        if st.button("Submit Claim", key=f"submit_{tid}"):
            if not claimant_name:
                st.warning("Enter your name")
            else:
                # If prize already awarded to someone, reject
                if claim_type in st.session_state.claimed_prizes and st.session_state.claimed_prizes[claim_type]:
                    st.error(f"{claim_type} already awarded to {st.session_state.claimed_prizes[claim_type]}")
                else:
                    cid = submit_claim(tid, claimant_name, claim_type)
                    st.success(f"Claim submitted (ID {cid}). Admin will validate.")
                    ok = check_prize_for_ticket(grid, called_set, claim_type)
                    if ok:
                        st.info("Quick check: claim appears VALID (final decision by admin).")
                    else:
                        st.info("Quick check: claim appears INVALID (admin will verify).")

# -------------------------
# CLAIMS QUEUE + ADMIN VALIDATION
# -------------------------
st.header("Claims Queue & History")
if st.session_state.claims_queue:
    st.dataframe(pd.DataFrame(st.session_state.claims_queue))
else:
    st.info("No claims submitted yet.")

if st.session_state.admin_unlocked:
    st.markdown("### Validate Claim")
    cid = st.number_input("Claim ID to process", min_value=0, step=1)
    action = st.selectbox("Action", ["Approve", "Reject"])
    approver = st.text_input("Approver name", value="Admin", key="approver_name")
    if st.button("Process Claim"):
        if cid <= 0:
            st.warning("Enter valid claim ID")
        else:
            recs = [r for r in st.session_state.claims_queue if r["id"] == cid]
            if not recs:
                st.error("Claim not found")
            else:
                rec = recs[0]
                valid_now = check_prize_for_ticket(st.session_state.tickets.get(rec["ticket_id"]), called_set, rec["claim_type"])
                approve = action == "Approve"
                if approve and not valid_now:
                    st.warning("Claim does not meet criteria right now. Approving will force award.")
                validate_claim(cid, approver, approve)
                if approve:
                    st.success(f"Claim {cid} approved. Awarded {rec['claim_type']} to {rec['ticket_id']}.")
                else:
                    st.info(f"Claim {cid} rejected.")

# -------------------------
# AWARDED PRIZES
# -------------------------
st.markdown("---")
st.header("Awarded Prizes")
if st.session_state.claimed_prizes:
    for p, t in st.session_state.claimed_prizes.items():
        st.write(f"🏅 {p}: {t if t else 'Eligible - admin review required'}")
else:
    st.info("No prizes awarded yet.")
