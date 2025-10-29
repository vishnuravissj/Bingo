# app.py
"""
Complete Tambola (Housie) Streamlit app (single file).
Features:
- Admin-key protected admin panel
- Ticket generation (N tickets)
- Exports: CSV of tickets, ZIP of PNG images
- Number drawing (random/manual)
- Dynamic prizes including "4 Corners"
- Player ticket view, claim submission
- Claims queue and admin approval/rejection with ticket grid shown for validation
- Uses st.session_state (no DB). Add DB persistence later if required.
"""

import streamlit as st
import random
import io
import zipfile
import pandas as pd
from datetime import datetime
from typing import Dict, List
from PIL import Image, ImageDraw, ImageFont

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="Tambola/BINGO (Housie)", layout="wide")
ADMIN_KEY = "VISHNU2025"  # change before sharing
DEFAULT_TICKET_COUNT = 200
TICKET_ROWS, TICKET_COLS = 3, 9
TICKET_IMG_W, TICKET_IMG_H = 540, 300
FONT_PATH = None  # optional path to .ttf

# -------------------------
# SESSION STATE INIT
# -------------------------
st.session_state.setdefault("tickets", {})                # Dict[str, List[List[int]]]
st.session_state.setdefault("called_numbers", [])         # List[int]
st.session_state.setdefault("claims_queue", [])           # List[dict]
st.session_state.setdefault("claimed_prizes", {})         # Dict[str, str] prize -> ticket_id
st.session_state.setdefault("loaded_ticket_id", None)     # persisted selected ticket
st.session_state.setdefault("admin_unlocked", False)
st.session_state.setdefault("last_draw", None)
st.session_state.setdefault("ticket_prefix", "T")
st.session_state.setdefault("prize_config", {
    "Early 5": True,
    "4 Corners": True,
    "Top Line": True,
    "Middle Line": True,
    "Bottom Line": True,
    "Full House": True
})

# -------------------------
# UTILS: Ticket generation
# -------------------------
COLUMN_RANGES = [(1,10),(11,20),(21,30),(31,40),(41,50),(51,60),(61,70),(71,80),(81,90)]

def generate_single_ticket() -> List[List[int]]:
    """Generate a valid 3x9 tambola ticket (each row has 5 numbers)."""
    while True:
        # choose counts per column summing to 15 with max 3 per column
        counts = [0]*9
        remaining = 15
        while remaining > 0:
            idx = random.randrange(9)
            if counts[idx] < 3:
                counts[idx] += 1
                remaining -= 1
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
        # distribute numbers into rows ensuring each row has 5 numbers
        grid = [[None]*9 for _ in range(3)]
        ok = True
        for col_idx, nums in enumerate(cols):
            rows = list(range(3))
            random.shuffle(rows)
            for n in nums:
                placed = False
                # put in row with least numbers first
                rows_sorted = sorted(rows, key=lambda r: sum(1 for x in grid[r] if x is not None))
                for r in rows_sorted:
                    if grid[r][col_idx] is None and sum(1 for x in grid[r] if x is not None) < 5:
                        grid[r][col_idx] = n
                        placed = True
                        break
                if not placed:
                    # fallback
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
            # replace None with 0 for consistency
            return [[(cell if cell is not None else 0) for cell in row] for row in grid]
        # else retry

def gen_unique_tickets(n: int, prefix: str = "T") -> Dict[str, List[List[int]]]:
    tickets = {}
    seen = set()
    attempts = 0
    while len(tickets) < n:
        attempts += 1
        if attempts > n * 2000:
            raise RuntimeError("Too many attempts to generate unique tickets.")
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
    """Return BytesIO PNG of ticket image."""
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
            x1 = x0 + cell_w
            y1 = y0 + cell_h
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
# PRIZE CHECK LOGIC
# -------------------------
def ticket_number_set(grid):
    return set(n for row in grid for n in row if n != 0)

def check_prize_for_ticket(grid, called_set, prize_name):
    """Return True if grid qualifies for prize_name given called_set (set of ints)."""
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
        # some ticket corners may be 0 (unlikely) — require non-zero and called
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
                # only award if prize still unclaimed
                if rec["claim_type"] not in st.session_state.claimed_prizes or not st.session_state.claimed_prizes[rec["claim_type"]]:
                    st.session_state.claimed_prizes[rec["claim_type"]] = rec["ticket_id"]
            return True
    return False

# -------------------------
# NUMBER DRAWING HELPERS
# -------------------------
def call_random_number():
    remaining = set(range(1,91)) - set(st.session_state.called_numbers)
    if not remaining:
        return None
    n = random.choice(list(remaining))
    st.session_state.called_numbers.append(n)
    st.session_state.last_draw = n
    return n

def call_manual_number(n: int):
    if n in st.session_state.called_numbers:
        return False
    st.session_state.called_numbers.append(n)
    st.session_state.last_draw = n
    return True

# -------------------------
# SIDEBAR: ADMIN UNLOCK
# -------------------------
with st.sidebar:
    st.title("Admin / Host")
    key = st.text_input("Admin Key", type="password")
    if st.button("Unlock Admin"):
        if key == ADMIN_KEY:
            st.session_state.admin_unlocked = True
            st.success("Admin unlocked ✅")
        else:
            st.error("Invalid admin key")

# -------------------------
# ADMIN PANEL
# -------------------------
if st.session_state.admin_unlocked:
    st.sidebar.markdown("---")
    st.sidebar.header("Admin Controls")

    # Prize configuration
    st.sidebar.subheader("Active Prizes")
    # ensure 4 Corners present
    if "4 Corners" not in st.session_state.prize_config:
        st.session_state.prize_config["4 Corners"] = True
    for pname in list(st.session_state.prize_config.keys()):
        st.session_state.prize_config[pname] = st.sidebar.checkbox(pname, value=st.session_state.prize_config[pname])

    st.sidebar.markdown("---")
    st.sidebar.header("Tickets")
    gen_n = st.sidebar.number_input("Generate how many tickets (overwrite)", min_value=1, max_value=2000, value=DEFAULT_TICKET_COUNT)
    prefix = st.sidebar.text_input("Ticket prefix", value=st.session_state.ticket_prefix)
    if st.sidebar.button("Generate Tickets (overwrite)"):
        st.session_state.tickets = gen_unique_tickets(gen_n, prefix)
        st.session_state.called_numbers = []
        st.session_state.claimed_prizes = {}
        st.session_state.claims_queue = []
        st.success(f"Generated {len(st.session_state.tickets)} tickets")

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
    st.sidebar.header("Number Caller")
    if st.sidebar.button("Call Random Number"):
        n = call_random_number()
        if n is None:
            st.sidebar.warning("All numbers called")
        else:
            st.sidebar.success(f"Called {n}")

    manual = st.sidebar.number_input("Manual call (1-90)", min_value=1, max_value=90, value=1)
    if st.sidebar.button("Call Manual Number"):
        ok = call_manual_number(manual)
        if not ok:
            st.sidebar.warning(f"{manual} already called")
        else:
            st.sidebar.success(f"Called {manual}")

    if st.sidebar.button("Reset Called Numbers (keep tickets)"):
        st.session_state.called_numbers = []
        st.sidebar.success("Called numbers cleared")

    st.sidebar.markdown("---")
    st.sidebar.header("Claims / Awarding")
    auto_assign = st.sidebar.checkbox("Auto-assign detected eligible prizes (mark eligible)", value=True)

# -------------------------
# MAIN UI - Header / Board
# -------------------------
st.title("🎯 Tambola / Housie — Hosted Game")
col_l, col_r = st.columns([2,1])
with col_r:
    st.metric("Numbers Called", f"{len(st.session_state.called_numbers)}/90")
    if st.session_state.last_draw:
        st.info(f"Last draw: {st.session_state.last_draw}")

st.header("Numbers Board")
called_sorted = sorted(st.session_state.called_numbers)
if called_sorted:
    cols = st.columns(9)
    for i, n in enumerate(called_sorted):
        with cols[i % 9]:
            st.button(str(n), key=f"num_{n}")
else:
    st.info("No numbers called yet")

# Auto-scan eligible tickets and mark prize as eligible (set to None if not yet assigned)
called_set = set(st.session_state.called_numbers)
for prize, enabled in st.session_state.prize_config.items():
    if not enabled:
        continue
    if prize in st.session_state.claimed_prizes and st.session_state.claimed_prizes[prize]:
        continue
    # if some ticket qualifies, mark eligible (None placeholder) so admin can see
    for tid, grid in st.session_state.tickets.items():
        if check_prize_for_ticket(grid, called_set, prize):
            st.session_state.claimed_prizes.setdefault(prize, None)
            break

st.markdown("---")

# -------------------------
# PLAYER: Load Ticket (persist) & Submit Claim
# -------------------------
st.header("Player Portal — Load Ticket & Submit Claim")
pid = st.text_input("Ticket ID", value=st.session_state.get("loaded_ticket_id") or "")

if st.button("Load Ticket"):
    if not pid:
        st.warning("Enter Ticket ID")
    else:
        if pid not in st.session_state.tickets:
            # create a ticket for the player if not present
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
        claimant = st.text_input("Your name", key=f"name_{tid}")
        active_prizes = [p for p, en in st.session_state.prize_config.items() if en]
        if not active_prizes:
            st.info("No active prizes configured by admin.")
        else:
            claim_type = st.selectbox("Claim type", active_prizes, key=f"claim_{tid}")
            if st.button("Submit Claim", key=f"submit_{tid}"):
                if not claimant:
                    st.warning("Enter your name")
                else:
                    # immediate quick-check of qualification
                    qualifies = check_prize_for_ticket(grid, called_set, claim_type)
                    cid = submit_claim(tid, claimant, claim_type)
                    st.success(f"Claim submitted (ID {cid}). Admin will validate.")
                    if qualifies:
                        st.info("Quick check: claim appears VALID (final admin verification required).")
                    else:
                        st.info("Quick check: claim appears INVALID (admin verification required).")

st.markdown("---")

# -------------------------
# CLAIMS QUEUE & ADMIN REVIEW
# -------------------------
st.header("Claims Queue & Admin Review")
if st.session_state.claims_queue:
    df = pd.DataFrame(st.session_state.claims_queue)
    st.dataframe(df[["id","ticket_id","claimant","claim_type","time","status"]])
else:
    st.info("No claims submitted yet.")

if st.session_state.admin_unlocked:
    st.markdown("### Review & Process Claims (Admin)")
    cid = st.number_input("Claim ID to process", min_value=0, step=1)
    action = st.selectbox("Action", ["Approve","Reject"])
    approver = st.text_input("Approver name", value="Admin", key="approver_name")
    if st.button("Process Claim"):
        if cid <= 0:
            st.warning("Enter a valid claim ID")
        else:
            recs = [r for r in st.session_state.claims_queue if r["id"] == cid]
            if not recs:
                st.error("Claim ID not found")
            else:
                rec = recs[0]
                # validate against current board
                valid_now = check_prize_for_ticket(st.session_state.tickets.get(rec["ticket_id"]), called_set, rec["claim_type"])
                approve = action == "Approve"
                if approve and not valid_now:
                    st.warning("Claim does not meet criteria based on current numbers. Approving will force award.")
                validate_claim(cid, approver, approve)
                if approve:
                    st.success(f"Claim {cid} approved. Prize '{rec['claim_type']}' awarded to {rec['ticket_id']}.")
                else:
                    st.info(f"Claim {cid} rejected.")

# -------------------------
# AWARDED / ELIGIBLE PRIZES (Admin-friendly display)
# -------------------------
st.markdown("---")
st.header("Awarded & Eligible Prizes")
# Show for each active prize: status, awarded ticket (if any), and list of qualifying tickets
for prize, enabled in st.session_state.prize_config.items():
    if not enabled:
        continue
    awarded_to = st.session_state.claimed_prizes.get(prize)
    # find all qualifying tickets given current called numbers
    qualifying = []
    for tid, grid in st.session_state.tickets.items():
        if check_prize_for_ticket(grid, called_set, prize):
            qualifying.append(tid)
    colA, colB = st.columns([3,2])
    with colA:
        st.write(f"### {prize}")
        if awarded_to:
            st.success(f"Awarded to: **{awarded_to}**")
        else:
            if qualifying:
                st.info(f"Eligible tickets ({len(qualifying)}): " + ", ".join(qualifying[:20]) + ("" if len(qualifying)<=20 else " ..."))
            else:
                st.write("No qualifying tickets yet.")
    with colB:
        # quick action: award to a selected qualifying ticket
        if qualifying and not awarded_to and st.session_state.admin_unlocked:
            pick = st.selectbox(f"Select ticket to award ({prize})", options=[""] + qualifying, key=f"pick_{prize}")
            if st.button(f"Award {prize} to selected", key=f"awardbtn_{prize}"):
                if pick:
                    st.session_state.claimed_prizes[prize] = pick
                    st.success(f"Awarded {prize} to {pick}")
                else:
                    st.warning("Select a ticket id first.")

st.caption("Tip: Approve claims from the Claims Queue. Use the Award controls above for direct awarding.")

# -------------------------
# TICKETS PREVIEW & EXPORT
# -------------------------
with st.expander("Tickets Preview & Export"):
    if st.session_state.tickets:
        df = tickets_to_dataframe(st.session_state.tickets)
        st.dataframe(df.head(500))
        csv = df.to_csv(index=False).encode()
        st.download_button("Download tickets CSV", csv, "tickets.csv", "text/csv")
        if st.button("Download All Ticket Images (ZIP)"):
            z = create_tickets_zip(st.session_state.tickets)
            st.download_button("Download ZIP", z, "tickets_images.zip", "application/zip")
    else:
        st.write("No tickets generated yet.")

st.caption("If you want persistence across restarts, I can add SQLite persistence next. Want me to do that?")
