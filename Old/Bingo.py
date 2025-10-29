# app.py
"""
Full Tambola (Housie) Streamlit app - single file.
Features:
- Dynamic prize control (enable/disable prizes at runtime)
- Generate tickets (3x9, 15 numbers per ticket)
- Draw numbers (random/manual)
- Player view (Ticket ID -> highlighted ticket)
- Claim submit & auto-validation
- Exports: CSV of tickets, ZIP of PNG ticket images
- Auto-detect winners (first valid wins)
Notes: For production persistence replace session_state with a DB (SQLite/Postgres).
"""

import streamlit as st
import random
import pandas as pd
import io
import zipfile
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="Tambola / Housie", layout="wide")
TITLE = "🎯 Tambola / Housie — Dynamic Prize Controller"
TICKET_ROWS = 3
TICKET_COLS = 9
DEFAULT_TICKET_COUNT = 200
TICKET_IMG_W, TICKET_IMG_H = 540, 300
FONT_PATH = None  # optional: set path to TTF for nicer output

# -------------------------
# SESSION STATE INIT
# -------------------------
if "called_numbers" not in st.session_state:
    st.session_state.called_numbers = set()
if "tickets" not in st.session_state:
    st.session_state.tickets = {}  # ticket_id -> grid (3x9 list)
if "claimed_prizes" not in st.session_state:
    st.session_state.claimed_prizes = {}  # prize_name -> ticket_id
if "claims" not in st.session_state:
    st.session_state.claims = []  # list of dicts {id, ticket_id, claimant, claim_type, validated, time}
if "game_started" not in st.session_state:
    st.session_state.game_started = False
if "last_draw" not in st.session_state:
    st.session_state.last_draw = None

# -------------------------
# UTILS: Ticket Generation
# -------------------------
COLUMN_RANGES = [(1,10),(11,20),(21,30),(31,40),(41,50),(51,60),(61,70),(71,80),(81,90)]

def generate_single_ticket():
    """Generates one valid 3x9 tambola ticket (each row exactly 5 numbers, columns within ranges)."""
    while True:
        # Choose counts per column summing to 15, each <=3
        counts = [0]*9
        remaining = 15
        while remaining > 0:
            idx = random.randrange(9)
            if counts[idx] < 3:
                counts[idx] += 1
                remaining -= 1
        # pick numbers for each column
        cols = []
        valid = True
        for i, cnt in enumerate(counts):
            lo, hi = COLUMN_RANGES[i]
            pool = list(range(lo, hi+1))
            if cnt > len(pool):
                valid = False; break
            cols.append(sorted(random.sample(pool, cnt)) if cnt>0 else [])
        if not valid:
            continue
        # distribute into rows; ensure each row has 5 numbers
        grid = [[None]*9 for _ in range(3)]
        ok = True
        for col_idx, nums in enumerate(cols):
            rows_for_col = random.sample([0,1,2], len(nums))
            for r, num in zip(rows_for_col, nums):
                grid[r][col_idx] = num
        # if any row doesn't have 5, try repairing by moving some from other rows
        if all(sum(1 for v in row if v is not None) == 5 for row in grid):
            return grid
        # else loop and regenerate (simpler) 

def ticket_to_flatlist(grid):
    return sorted([n for row in grid for n in row if n is not None])

def gen_unique_tickets(n, prefix="T"):
    tickets = {}
    seen = set()
    attempts = 0
    while len(tickets) < n:
        attempts += 1
        if attempts > n * 3000:
            raise RuntimeError("Too many attempts generating unique tickets. Try fewer tickets.")
        grid = generate_single_ticket()
        key = tuple(ticket_to_flatlist(grid))
        if key in seen:
            continue
        seen.add(key)
        tid = f"{prefix}{len(tickets)+1:04d}"
        tickets[tid] = grid
    return tickets

# -------------------------
# UTILS: Render ticket image
# -------------------------
def render_ticket_image(grid, ticket_id):
    img = Image.new("RGB", (TICKET_IMG_W, TICKET_IMG_H), "white")
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

    # header
    header_text = f"Ticket {ticket_id}"
    try:
        bbox = draw.textbbox((0,0), header_text, font=header_font)
        text_w = bbox[2]-bbox[0]
    except AttributeError:
        text_w, _ = draw.textsize(header_text, font=header_font)
    draw.text(((TICKET_IMG_W-text_w)/2, 8), header_text, font=header_font, fill=(0,0,0))

    left, top = 12, 40
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
                    w = bbox[2]-bbox[0]; h = bbox[3]-bbox[1]
                except AttributeError:
                    w,h = draw.textsize(txt, font=cell_font)
                draw.text((x0 + (cell_w-w)/2, y0 + (cell_h-h)/2), txt, font=cell_font, fill=(0,0,0))
    # return bytes
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    bio.seek(0)
    return bio

# -------------------------
# UTILS: Export functions
# -------------------------
def tickets_to_dataframe(tickets):
    rows = []
    for tid, grid in tickets.items():
        nums = [str(n) for row in grid for n in row if n is not None]
        rows.append({"ticket_id": tid, "numbers": ",".join(nums)})
    return pd.DataFrame(rows)

def create_zip_of_images(tickets):
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w") as z:
        for tid, grid in tickets.items():
            img_b = render_ticket_image(grid, tid)
            z.writestr(f"{tid}.png", img_b.read())
    mem.seek(0)
    return mem

# -------------------------
# UTILS: Claim & validation
# -------------------------
def submit_claim(ticket_id, claimant, claim_type):
    cid = len(st.session_state.claims) + 1
    rec = {
        "id": cid,
        "ticket_id": ticket_id,
        "claimant": claimant,
        "claim_type": claim_type,
        "validated": None,
        "time": datetime.utcnow().isoformat()
    }
    st.session_state.claims.insert(0, rec)
    return cid

def validate_claim_record(cid, valid):
    for rec in st.session_state.claims:
        if rec["id"] == cid:
            rec["validated"] = bool(valid)
            return True
    return False

def check_claim_validity_with_called(grid, called_set, claim_type):
    all_nums = set([n for row in grid for n in row if n is not None])
    if claim_type == "Early 5":
        return len(all_nums & called_set) >= 5
    if claim_type == "Top Line":
        top = set(n for n in grid[0] if n is not None)
        return top.issubset(called_set)
    if claim_type == "Middle Line":
        mid = set(n for n in grid[1] if n is not None)
        return mid.issubset(called_set)
    if claim_type == "Bottom Line":
        bot = set(n for n in grid[2] if n is not None)
        return bot.issubset(called_set)
    if claim_type == "Full House":
        return all_nums.issubset(called_set)
    return False

# -------------------------
# UI: Title & sidebar controls (dynamic prize config)
# -------------------------
st.title(TITLE)
st.sidebar.header("Admin Controls")

# Prize toggles (dynamic)
st.sidebar.subheader("Prize Configuration")
prize_enable = {
    "Early 5": st.sidebar.checkbox("Enable Early 5", True),
    "Top Line": st.sidebar.checkbox("Enable Top Line", True),
    "Middle Line": st.sidebar.checkbox("Enable Middle Line", True),
    "Bottom Line": st.sidebar.checkbox("Enable Bottom Line", True),
    "Full House": st.sidebar.checkbox("Enable Full House", True)
}

# generate tickets input
st.sidebar.markdown("---")
ticket_count = st.sidebar.number_input("Total tickets to generate", min_value=1, max_value=2000, value=DEFAULT_TICKET_COUNT, step=1)
ticket_prefix = st.sidebar.text_input("Ticket prefix", "T")

if st.sidebar.button("Generate new tickets (overwrite)"):
    st.session_state.tickets = gen_unique_tickets(ticket_count, prefix=ticket_prefix)
    st.session_state.called_numbers = set()
    st.session_state.claimed_prizes = {}
    st.session_state.claims = []
    st.session_state.game_started = False
    st.sidebar.success(f"Generated {len(st.session_state.tickets)} tickets.")

if st.sidebar.button("Append tickets"):
    new = gen_unique_tickets(ticket_count, prefix=ticket_prefix)
    # merge ensuring unique ids
    max_existing = len(st.session_state.tickets)
    for i, (tid, grid) in enumerate(new.items()):
        new_id = f"{ticket_prefix}{max_existing + i + 1:04d}"
        st.session_state.tickets[new_id] = grid
    st.sidebar.success(f"Appended {len(new)} tickets.")

# drawing controls
st.sidebar.markdown("---")
st.sidebar.subheader("Number Caller")
if st.sidebar.button("Start/Reset Game (keep tickets)"):
    st.session_state.called_numbers = set()
    st.session_state.claimed_prizes = {}
    st.session_state.claims = []
    st.session_state.game_started = True
    st.sidebar.success("Game started and called numbers reset.")

manual_num = st.sidebar.number_input("Manual call number (1-90)", min_value=1, max_value=90, value=1, step=1)
if st.sidebar.button("Call Manual Number"):
    if manual_num in st.session_state.called_numbers:
        st.sidebar.warning(f"{manual_num} already called.")
    else:
        st.session_state.called_numbers.add(manual_num)
        st.session_state.last_draw = manual_num
        st.sidebar.success(f"Called {manual_num}")

if st.sidebar.button("Call Random Number"):
    remaining = set(range(1,91)) - st.session_state.called_numbers
    if not remaining:
        st.sidebar.warning("All numbers called already.")
    else:
        n = random.choice(list(remaining))
        st.session_state.called_numbers.add(n)
        st.session_state.last_draw = n
        st.sidebar.success(f"Random Called: {n}")

if st.sidebar.button("Reset All (clear tickets & state)"):
    st.session_state.tickets = {}
    st.session_state.called_numbers = set()
    st.session_state.claimed_prizes = {}
    st.session_state.claims = []
    st.sidebar.success("Cleared tickets and state.")

# export
st.sidebar.markdown("---")
st.sidebar.subheader("Exports")
if st.sidebar.button("Export tickets CSV"):
    if not st.session_state.tickets:
        st.sidebar.warning("No tickets to export.")
    else:
        df = tickets_to_dataframe(st.session_state.tickets)
        csv = df.to_csv(index=False).encode()
        st.sidebar.download_button("Download tickets CSV", csv, "tickets.csv", "text/csv")
if st.sidebar.button("Export ticket images ZIP"):
    if not st.session_state.tickets:
        st.sidebar.warning("No tickets to export.")
    else:
        z = create_zip_of_images(st.session_state.tickets)
        st.sidebar.download_button("Download Tickets ZIP", z, "tickets_images.zip", "application/zip")

# -------------------------
# MAIN UI: Top area
# -------------------------
col1, col2 = st.columns([2,1])
with col1:
    st.header("Game Console")
    st.write("Numbers called:")
    st.write(sorted(list(st.session_state.called_numbers)))
    if st.session_state.last_draw is not None:
        st.info(f"Last drawn number: {st.session_state.last_draw}")

    # Draw button in main UI too
    if st.button("Draw Next (Main)"):
        remaining = set(range(1,91)) - st.session_state.called_numbers
        if not remaining:
            st.warning("All numbers called.")
        else:
            n = random.choice(list(remaining))
            st.session_state.called_numbers.add(n)
            st.session_state.last_draw = n
            st.success(f"Number Drawn: {n}")

with col2:
    st.header("Winners / Prizes")
    # auto-detect winners based on enabled prizes and first-to-find
    # check all tickets; assign winners for enabled prizes only if not already assigned
    called = set(st.session_state.called_numbers)
    for prize, enabled in prize_enable.items():
        if not enabled:
            continue
        if prize in st.session_state.claimed_prizes:
            continue
        # scan tickets for first qualifying ticket
        for tid, grid in st.session_state.tickets.items():
            if check_claim_validity_with_called(grid, called, prize):
                st.session_state.claimed_prizes[prize] = tid
                st.success(f"Prize '{prize}' auto-assigned to {tid}")
                break
    if st.session_state.claimed_prizes:
        for p, winner in st.session_state.claimed_prizes.items():
            st.write(f"🏅 {p}: {winner}")
    else:
        st.info("No prizes claimed yet.")

st.markdown("---")

# -------------------------
# Player Portal & Claim UI
# -------------------------
st.header("Player Portal")
st.write("Enter your Ticket ID to view your ticket, see highlighted called numbers, and submit a claim.")

pid = st.text_input("Ticket ID (case-sensitive)", "")
if st.button("Load Ticket"):
    if not pid:
        st.warning("Enter Ticket ID.")
    else:
        if pid not in st.session_state.tickets:
            st.error("Ticket ID not found. Contact host.")
        else:
            grid = st.session_state.tickets[pid]
            # render ticket image with highlights for called numbers
            img_b = render_ticket_image(grid, pid)
            # re-render to highlight called numbers: draw overlay
            img = Image.open(img_b)
            draw = ImageDraw.Draw(img)
            try:
                if FONT_PATH:
                    cell_font = ImageFont.truetype(FONT_PATH, 16)
                else:
                    cell_font = ImageFont.load_default()
            except:
                cell_font = ImageFont.load_default()
            left, top = 12, 40
            cell_w = (TICKET_IMG_W - left*2)//TICKET_COLS
            cell_h = (TICKET_IMG_H - top - 10)//TICKET_ROWS
            called = set(st.session_state.called_numbers)
            for r in range(TICKET_ROWS):
                for c in range(TICKET_COLS):
                    val = grid[r][c]
                    if val is not None and val in called:
                        x0 = left + c*cell_w + 2
                        y0 = top + r*cell_h + 2
                        x1 = x0 + cell_w - 4
                        y1 = y0 + cell_h - 4
                        draw.rectangle([x0,y0,x1,y1], fill=(200,255,200))
                        txt = str(val)
                        try:
                            bbox = draw.textbbox((0,0), txt, font=cell_font)
                            w = bbox[2]-bbox[0]; h = bbox[3]-bbox[1]
                        except AttributeError:
                            w,h = draw.textsize(txt, font=cell_font)
                        draw.text((x0 + (cell_w - 4 - w)/2, y0 + (cell_h - 4 - h)/2), txt, font=cell_font, fill=(0,0,0))
            bio = io.BytesIO(); img.save(bio, format="PNG"); bio.seek(0)
            st.image(bio)

            st.write("Numbers called so far:", sorted(list(called)))
            st.markdown("### Submit Claim")
            claimant = st.text_input("Your name", key=f"name_{pid}")
            claim_type = st.selectbox("Claim Type", ["Early 5", "Top Line", "Middle Line", "Bottom Line", "Full House"], key=f"claim_{pid}")
            if st.button("Submit Claim", key=f"submit_{pid}"):
                cid = submit_claim(pid, claimant, claim_type)
                st.success(f"Claim submitted (ID {cid}). Host will validate.")
                # quick auto-check (informational only)
                ok = check_claim_validity_with_called(grid, called, claim_type)
                if ok:
                    st.info("Claim seems VALID against current called numbers (host validation still required).")
                else:
                    st.info("Claim appears INVALID with current called numbers.")

st.markdown("---")

# -------------------------
# Claims review area (host)
# -------------------------
st.header("Claims Log (Host)")
if st.session_state.claims:
    dfc = pd.DataFrame(st.session_state.claims)
    st.dataframe(dfc)
    st.write("Validate a claim below:")
    cid_to_val = st.number_input("Claim ID to validate", min_value=0, step=1, value=0)
    if st.button("Validate Claim (Host)"):
        if cid_to_val <= 0:
            st.warning("Enter a claim ID.")
        else:
            recs = [c for c in st.session_state.claims if c["id"] == cid_to_val]
            if not recs:
                st.error("Claim ID not found.")
            else:
                rec = recs[0]
                tid = rec["ticket_id"]
                grid = st.session_state.tickets.get(tid)
                if not grid:
                    st.error("Ticket not found.")
                else:
                    valid = check_claim_validity_with_called(grid, set(st.session_state.called_numbers), rec["claim_type"])
                    validate_claim_record(cid_to_val, valid)
                    if valid:
                        st.success(f"Claim {cid_to_val} VALID. Awarding prize: {rec['claim_type']} to {tid}")
                        if prize_enable.get(rec["claim_type"], False) and rec["claim_type"] not in st.session_state.claimed_prizes:
                            st.session_state.claimed_prizes[rec["claim_type"]] = tid
                    else:
                        st.error(f"Claim {cid_to_val} NOT valid.")
else:
    st.info("No claims submitted yet.")

# -------------------------
# Show tickets table & quick download
# -------------------------
with st.expander("📋 Tickets (preview & download)"):
    if st.session_state.tickets:
        df = tickets_to_dataframe(st.session_state.tickets)
        st.dataframe(df.head(500))
        csv = df.to_csv(index=False).encode()
        st.download_button("Download tickets CSV", csv, "tickets.csv", "text/csv")
    else:
        st.write("No tickets generated yet.")

st.markdown("---")
st.caption("Built for quick setup. For production: add DB persistence, authentication, and rate-limited number draws.")
