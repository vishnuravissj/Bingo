# app.py
"""
Tambola (Housie) with Real-time Claim System
- Admin protected by ADMIN_KEY
- Ticket generation (3x9, 15 numbers)
- Number calling (random/manual) by Admin
- Player ticket view + Claim submission
- Admin claim queue: Validate (approve) / Reject
- Prize definitions dynamic (enable/disable)
- Exports: tickets CSV + ZIP of PNGs
"""

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
st.set_page_config(page_title="Tambola - Claims & Admin", layout="wide")
ADMIN_KEY = "vishnu@tambola"  # change to secure key before production
DEFAULT_TICKET_COUNT = 200
TICKET_ROWS, TICKET_COLS = 3, 9
TICKET_IMG_W, TICKET_IMG_H = 540, 300
FONT_PATH = None  # optional: path to a .ttf for nicer images

# -------------------------
# ✅ SESSION STATE INIT (CLEAN VERSION)
# -------------------------
st.session_state.setdefault("tickets", {})  # type: Dict[str, List[List[int]]]
st.session_state.setdefault("called_numbers", [])  # type: List[int]
st.session_state.setdefault("claimed_prizes", {})  # type: Dict[str, str]  # prize -> ticket_id
st.session_state.setdefault("claims_queue", [])  # type: List[dict]  # latest first
st.session_state.setdefault("game_started", False)
st.session_state.setdefault("last_draw", None)
st.session_state.setdefault("prize_config", {
    "Early 5": True,
    "Top Line": True,
    "Middle Line": True,
    "Bottom Line": True,
    "Full House": True
})
st.session_state.setdefault("auto_assign_on_validate", True)

# -------------------------
# UTILS: Ticket generation (reasonable approach)
# -------------------------
COLUMN_RANGES = [(1,10),(11,20),(21,30),(31,40),(41,50),(51,60),(61,70),(71,80),(81,90)]

def generate_single_ticket():
    # Robust generator: choose counts per column summing to 15 with max 3 per column,
    # distribute into rows ensuring each row has 5 numbers.
    while True:
        counts = [0]*9
        remaining = 15
        while remaining > 0:
            idx = random.randrange(9)
            if counts[idx] < 3:
                counts[idx] += 1
                remaining -= 1
        cols = []
        valid = True
        for i,cnt in enumerate(counts):
            lo,hi = COLUMN_RANGES[i]
            pool = list(range(lo, hi+1))
            if cnt > len(pool):
                valid = False; break
            cols.append(sorted(random.sample(pool, cnt)) if cnt>0 else [])
        if not valid:
            continue
        grid = [[None]*9 for _ in range(3)]
        for col_idx, numbers in enumerate(cols):
            rows = list(range(3))
            random.shuffle(rows)
            for n in numbers:
                # place in a row with <5 numbers currently
                placed=False
                rows_sorted = sorted(rows, key=lambda r: sum(1 for x in grid[r] if x is not None))
                for r in rows_sorted:
                    if grid[r][col_idx] is None and sum(1 for x in grid[r] if x is not None) < 5:
                        grid[r][col_idx] = n
                        placed = True
                        break
                if not placed:
                    # fallback - try any
                    for r in range(3):
                        if grid[r][col_idx] is None and sum(1 for x in grid[r] if x is not None) < 5:
                            grid[r][col_idx] = n
                            placed = True
                            break
                if not placed:
                    placed=False
                    break
            if not placed:
                break
        if all(sum(1 for x in row if x is not None) == 5 for row in grid):
            return grid
        # else retry

def gen_unique_tickets(n, prefix="T"):
    tickets = {}
    seen = set()
    attempts=0
    while len(tickets) < n:
        attempts+=1
        if attempts > n*2000:
            raise RuntimeError("Too many attempts generating tickets; reduce N")
        grid = generate_single_ticket()
        key = tuple(sorted([x for row in grid for x in row if x is not None]))
        if key in seen:
            continue
        seen.add(key)
        tid = f"{prefix}{len(tickets)+1:04d}"
        tickets[tid] = grid
    return tickets

# -------------------------
# UTILS: Render ticket image (Pillow modern API safe)
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
    left = 10; top = 40
    cell_w = (TICKET_IMG_W - left*2)//TICKET_COLS
    cell_h = (TICKET_IMG_H - top - 10)//TICKET_ROWS
    for r in range(TICKET_ROWS):
        for c in range(TICKET_COLS):
            x0 = left + c*cell_w
            y0 = top + r*cell_h
            x1 = x0 + cell_w
            y1 = y0 + cell_h
            draw.rectangle([x0,y0,x1,y1], outline=(0,0,0), width=1)
            val = grid[r][c]
            if val is not None:
                txt = str(val)
                try:
                    bbox = draw.textbbox((0,0), txt, font=cell_font)
                    w,h = bbox[2]-bbox[0], bbox[3]-bbox[1]
                except AttributeError:
                    w,h = draw.textsize(txt, font=cell_font)
                tx = x0 + (cell_w - w)//2
                ty = y0 + (cell_h - h)//2
                draw.text((tx,ty), txt, font=cell_font, fill=(0,0,0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

# -------------------------
# UTILS: Export CSV & ZIP
# -------------------------
def tickets_to_dataframe(tickets: Dict[str, List[List[int]]]) -> pd.DataFrame:
    rows=[]
    for tid, grid in tickets.items():
        nums = [str(n) for row in grid for n in row if n is not None]
        rows.append({"ticket_id": tid, "numbers": ",".join(nums)})
    return pd.DataFrame(rows)

def create_tickets_zip(tickets):
    mem=io.BytesIO()
    with zipfile.ZipFile(mem, "w") as zf:
        for tid, grid in tickets.items():
            img_b = render_ticket_image(grid, tid)
            zf.writestr(f"{tid}.png", img_b.read())
    mem.seek(0)
    return mem

# -------------------------
# PRIZE VALIDATION LOGIC
# -------------------------
def ticket_number_set(grid):
    return set(n for row in grid for n in row if n is not None)

def check_prize_for_ticket(grid, called_set, prize_name):
    if prize_name == "Early 5":
        return len(ticket_number_set(grid) & called_set) >= 5
    if prize_name == "Top Line":
        top = set(n for n in grid[0] if n is not None)
        return top.issubset(called_set)
    if prize_name == "Middle Line":
        mid = set(n for n in grid[1] if n is not None)
        return mid.issubset(called_set)
    if prize_name == "Bottom Line":
        bot = set(n for n in grid[2] if n is not None)
        return bot.issubset(called_set)
    if prize_name == "Full House":
        return ticket_number_set(grid).issubset(called_set)
    return False

# -------------------------
# CLAIM SUBMISSION & VALIDATION
# -------------------------
def submit_claim(ticket_id: str, claimant: str, claim_type: str) -> int:
    # create claim record and push to front (latest first)
    cid = len(st.session_state.claims_queue) + 1
    rec = {
        "id": cid,
        "ticket_id": ticket_id,
        "claimant": claimant,
        "claim_type": claim_type,
        "time": datetime.utcnow().isoformat(),
        "status": "pending",  # pending / approved / rejected
        "validated_by": None,
        "validated_at": None
    }
    st.session_state.claims_queue.insert(0, rec)
    return cid

def validate_claim(cid:int, approver:str, approve:bool):
    for rec in st.session_state.claims_queue:
        if rec["id"] == cid:
            rec["status"] = "approved" if approve else "rejected"
            rec["validated_by"] = approver
            rec["validated_at"] = datetime.utcnow().isoformat()
            # if approved and prize still available, assign prize to ticket
            if approve and rec["claim_type"] in st.session_state.prize_config and st.session_state.prize_config[rec["claim_type"]]:
                if rec["claim_type"] not in st.session_state.claimed_prizes:
                    st.session_state.claimed_prizes[rec["claim_type"]] = rec["ticket_id"]
            return True
    return False

# -------------------------
# UI: Admin authentication
# -------------------------
st.sidebar.header("Admin Authentication")
admin_input = st.sidebar.text_input("Enter Admin Key", type="password")
is_admin = admin_input == ADMIN_KEY
if is_admin:
    st.sidebar.success("Admin access granted ✅")
else:
    if admin_input:
        st.sidebar.warning("Invalid admin key")

# -------------------------
# UI: Admin Panel (secure)
# -------------------------
if is_admin:
    st.sidebar.markdown("---")
    st.sidebar.header("Admin Controls")

    # Prize toggles
    st.sidebar.subheader("Active Prizes")
    for pname in list(st.session_state.prize_config.keys()):
        st.session_state.prize_config[pname] = st.sidebar.checkbox(pname, value=st.session_state.prize_config[pname])

    st.sidebar.markdown("---")
    # ticket operations
    gen_count = st.sidebar.number_input("Number of tickets to generate", min_value=1, max_value=2000, value=DEFAULT_TICKET_COUNT, step=1)
    prefix = st.sidebar.text_input("Ticket prefix", value="T")

    if st.sidebar.button("Generate Tickets (overwrite)"):
        st.session_state.tickets = gen_unique_tickets(gen_count, prefix)
        st.session_state.called_numbers = []
        st.session_state.claimed_prizes = {}
        st.session_state.claims_queue = []
        st.sidebar.success(f"Generated {len(st.session_state.tickets)} tickets")

    if st.sidebar.button("Export Tickets CSV"):
        if not st.session_state.tickets:
            st.sidebar.warning("No tickets")
        else:
            df = tickets_to_dataframe(st.session_state.tickets)
            csv = df.to_csv(index=False).encode()
            st.sidebar.download_button("Download CSV", csv, "tickets.csv", "text/csv")

    if st.sidebar.button("Export Ticket Images (ZIP)"):
        if not st.session_state.tickets:
            st.sidebar.warning("No tickets")
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

    manual = st.sidebar.number_input("Manual call (1-90)", min_value=1, max_value=90, value=1, step=1)
    if st.sidebar.button("Call Manual Number"):
        if manual in st.session_state.called_numbers:
            st.sidebar.warning(f"{manual} already called")
        else:
            st.session_state.called_numbers.append(manual)
            st.session_state.last_draw = manual
            st.sidebar.success(f"Called {manual}")

    if st.sidebar.button("Reset Called Numbers (keep tickets)"):
        st.session_state.called_numbers = []
        st.sidebar.success("Called numbers cleared")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Claim Handling")
    st.session_state.auto_assign_on_validate = st.sidebar.checkbox("Auto-assign prize when claim approved", value=st.session_state.auto_assign_on_validate)

# -------------------------
# MAIN UI: Title + status
# -------------------------
st.title("🎯 Tambola / Housie — Live (Claims Enabled)")
st.markdown("Use admin key in sidebar to access admin controls.")

col_a, col_b = st.columns([2,1])

with col_b:
    st.metric("Numbers Called", f"{len(st.session_state.called_numbers)}/90")
    if st.session_state.last_draw:
        st.info(f"Last draw: {st.session_state.last_draw}")

# -------------------------
# Display called numbers (grid)
# -------------------------
st.header("Numbers Board")
called_sorted = sorted(st.session_state.called_numbers)
if called_sorted:
    cols = st.columns(9)
    for i, num in enumerate(called_sorted):
        with cols[i % 9]:
            st.button(str(num), key=f"num_{num}")
else:
    st.info("No numbers called yet")

# -------------------------
# Auto-check & assign winners (optional)
# -------------------------
# If admin enabled auto-assign on validate, we still rely on admin to approve claims.
# Here we provide an auto-scan that can pre-populate winners (only if prize not yet claimed).
called_set = set(st.session_state.called_numbers)
for prize, enabled in st.session_state.prize_config.items():
    if not enabled:
        continue
    if prize in st.session_state.claimed_prizes:
        continue
    # scan tickets
    for tid, grid in st.session_state.tickets.items():
        if check_prize_for_ticket(grid, called_set, prize):
            # auto-assign if admin set auto-assign mode (use carefully)
            # For safety, do not auto-approve claim records; instead mark prize available for admin
            st.session_state.claimed_prizes.setdefault(prize, None)
            # leave None to indicate eligible ticket(s) exist; admin should validate
            break

# -------------------------
# Player Portal: view ticket and submit claim
# -------------------------
st.header("Player Portal — View Ticket & Submit Claim")
st.write("Enter your Ticket ID (given by admin) and submit claim when you have a winning pattern.")

pid = st.text_input("Ticket ID", key="player_ticket_input")
if st.button("Load Ticket", key="load_ticket"):
    if not pid:
        st.warning("Enter Ticket ID")
    elif pid not in st.session_state.tickets:
        st.error("Ticket ID not found. Ask admin.")
    else:
        grid = st.session_state.tickets[pid]
        # render ticket with highlights for called numbers
        img_b = render_ticket_image(grid, pid)
        img = Image.open(img_b)
        draw = ImageDraw.Draw(img)
        try:
            cell_font = ImageFont.truetype(FONT_PATH, 14) if FONT_PATH else ImageFont.load_default()
        except Exception:
            cell_font = ImageFont.load_default()
        left, top = 10, 40
        cw = (TICKET_IMG_W - left*2)//TICKET_COLS
        ch = (TICKET_IMG_H - top - 10)//TICKET_ROWS
        called = set(st.session_state.called_numbers)
        for r in range(TICKET_ROWS):
            for c in range(TICKET_COLS):
                val = grid[r][c]
                if val is not None and val in called:
                    x0 = left + c*cw + 2
                    y0 = top + r*ch + 2
                    x1 = x0 + cw - 4
                    y1 = y0 + ch - 4
                    draw.rectangle([x0,y0,x1,y1], fill=(180,255,200))
                    txt = str(val)
                    try:
                        bbox = draw.textbbox((0,0), txt, font=cell_font)
                        w = bbox[2]-bbox[0]; h = bbox[3]-bbox[1]
                    except AttributeError:
                        w,h = draw.textsize(txt, font=cell_font)
                    draw.text((x0+(cw-4-w)/2, y0+(ch-4-h)/2), txt, font=cell_font, fill=(0,0,0))
        buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
        st.image(buf)

        st.write("Called numbers so far:", sorted(list(called)))
        st.markdown("### Submit Claim")
        claimant_name = st.text_input("Your name", key=f"name_{pid}")
        claim_type = st.selectbox("Claim type", list(st.session_state.prize_config.keys()), key=f"claimtype_{pid}")
        if st.button("Submit Claim", key=f"submit_{pid}"):
            if not claimant_name:
                st.warning("Enter your name")
            else:
                # quick local validation: if flag false, still allow submission but admin will reject
                cid = submit_claim(pid, claimant_name, claim_type)
                st.success(f"Claim submitted (ID {cid}). Admin will validate soon.")
                # optional quick-check feedback
                ok = check_prize_for_ticket(grid, called_set, claim_type)
                if ok:
                    st.info("Claim appears VALID based on current called numbers (final decision by admin).")
                else:
                    st.info("Claim appears INVALID at this moment (admin will verify).")

# -------------------------
# Claims Queue (Admin view)
# -------------------------
st.header("Claims Queue & History")
if st.session_state.claims_queue:
    df = pd.DataFrame(st.session_state.claims_queue)
    st.dataframe(df[["id","ticket_id","claimant","claim_type","time","status"]])
else:
    st.info("No claims submitted yet.")

if is_admin:
    st.markdown("### Validate Claims (Admin)")
    cid = st.number_input("Enter Claim ID to validate", min_value=0, step=1)
    approve = st.selectbox("Approve or Reject", ["Approve", "Reject"])
    approver_name = st.text_input("Approver name", value="Admin", key="approver_name")
    if st.button("Process Claim"):
        if cid <= 0:
            st.warning("Enter a valid claim ID")
        else:
            # find claim
            recs = [r for r in st.session_state.claims_queue if r["id"] == cid]
            if not recs:
                st.error("Claim ID not found")
            else:
                rec = recs[0]
                # validate against current numbers too
                valid_now = check_prize_for_ticket(st.session_state.tickets.get(rec["ticket_id"]), called_set, rec["claim_type"])
                approved = approve == "Approve"
                # if admin approves but validation fails, warn but allow override
                if approved and not valid_now:
                    st.warning("Claim does not meet criteria based on current numbers. Approving will force award.")
                validate_claim(cid, approver_name, approved)
                if approved:
                    # assign prize if not already claimed
                    if rec["claim_type"] not in st.session_state.claimed_prizes or st.session_state.claimed_prizes[rec["claim_type"]] is None:
                        st.session_state.claimed_prizes[rec["claim_type"]] = rec["ticket_id"]
                    st.success(f"Claim {cid} approved. Prize '{rec['claim_type']}' awarded to {rec['ticket_id']}.")
                else:
                    st.info(f"Claim {cid} rejected.")

# -------------------------
# Show current awarded prizes
# -------------------------
st.markdown("---")
st.header("Awarded Prizes")
if st.session_state.claimed_prizes:
    for p, t in st.session_state.claimed_prizes.items():
        st.write(f"🏅 {p}: {t if t else 'Eligible - admin review required'}")
else:
    st.info("No prizes awarded yet.")

# -------------------------
# Tickets preview & export
# -------------------------
with st.expander("Tickets Preview & Export"):
    if st.session_state.tickets:
        df = tickets_to_dataframe(st.session_state.tickets)
        st.dataframe(df.head(500))
        csv = df.to_csv(index=False).encode()
        st.download_button("Download tickets CSV", csv, "tickets.csv", "text/csv")
        if st.button("Download Tickets ZIP"):
            z = create_tickets_zip(st.session_state.tickets)
            st.download_button("Download ZIP", z, "tickets_images.zip", "application/zip")
    else:
        st.write("No tickets generated yet. Admin should create tickets.")

st.caption("Tip: for production persistence, I can add SQLite/Postgres backing so tickets & claims survive restarts. Ask me to add DB persistence if you want.")
