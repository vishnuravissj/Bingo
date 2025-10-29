# app.py
"""
Complete Streamlit Tambola / Housie app (single-file).
Features:
- Generate N unique tickets (default 200)
- Store tickets, calls, claims in local SQLite DB (tambola.db)
- Host dashboard: call numbers, export tickets (CSV/Excel/ZIP of PNGs), validate claims
- Player view: enter Ticket ID, view highlighted ticket, submit claims
- Auto-validation & win tracking: Early Five, Top/Middle/Bottom Line, Full House
Notes:
- Designed to run locally or on Streamlit Cloud.
- requirements.txt must include: streamlit, pandas, xlsxwriter, pillow, numpy
"""

import streamlit as st
import sqlite3
import json
import random
import pandas as pd
import numpy as np
from io import BytesIO
from zipfile import ZipFile
from PIL import Image, ImageDraw, ImageFont
import os
from datetime import datetime

# --------------------
# CONFIG
# --------------------
DB_PATH = "tambola.db"
DEFAULT_TICKET_COUNT = 200
TICKET_WIDTH_PX = 540
TICKET_HEIGHT_PX = 300
FONT_PATH = None  # optional: set to a .ttf path for nicer images
COLUMN_RANGES = [(1,10),(11,20),(21,30),(31,40),(41,50),(51,60),(61,70),(71,80),(81,90)]

# --------------------
# DB Helpers
# --------------------
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS tickets (
                    id TEXT PRIMARY KEY,
                    numbers_json TEXT,
                    created_at TEXT
                )""")
    c.execute("""CREATE TABLE IF NOT EXISTS game_state (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )""")
    c.execute("""CREATE TABLE IF NOT EXISTS claims (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket_id TEXT,
                    claim_type TEXT,
                    claimant_name TEXT,
                    numbers_json TEXT,
                    validated INTEGER,
                    validated_at TEXT,
                    created_at TEXT
                )""")
    conn.commit()
    return conn

conn = init_db()

# --------------------
# Ticket Generation (robust)
# --------------------
def generate_single_ticket():
    # generate counts per column summing to 15, each <=3
    while True:
        counts = [0]*9
        remaining = 15
        # Greedy: add 1 randomly until remaining 0 while limiting to 3
        while remaining > 0:
            idx = random.randrange(9)
            if counts[idx] < 3:
                counts[idx] += 1
                remaining -= 1
        # Now build numbers per column
        grid_cols = []
        valid = True
        for i, cnt in enumerate(counts):
            lo, hi = COLUMN_RANGES[i]
            choices = list(range(lo, hi+1))
            if cnt > len(choices):
                valid = False; break
            selected = random.sample(choices, cnt) if cnt>0 else []
            selected.sort()
            grid_cols.append(selected)
        if not valid:
            continue
        # distribute into 3 rows ensuring each row has 5 numbers
        grid = [[None]*9 for _ in range(3)]
        for col_idx, numbers in enumerate(grid_cols):
            # pick rows for each number (distinct rows)
            row_choices = [0,1,2]
            random.shuffle(row_choices)
            for j, num in enumerate(numbers):
                placed = False
                # attempt to put into least-filled valid row
                row_order = sorted(row_choices, key=lambda r: sum(1 for x in grid[r] if x is not None))
                for r in row_order:
                    if grid[r][col_idx] is None and sum(1 for x in grid[r] if x is not None) < 5:
                        grid[r][col_idx] = num
                        placed = True
                        break
                if not placed:
                    # fallback: try any row
                    for r in range(3):
                        if grid[r][col_idx] is None and sum(1 for x in grid[r] if x is not None) < 5:
                            grid[r][col_idx] = num
                            placed = True
                            break
                if not placed:
                    placed = False
                    break
            if not placed:
                break
        # validate rows have exactly 5 numbers
        if all(sum(1 for x in grid[r] if x is not None) == 5 for r in range(3)):
            return grid
        # else retry

def ticket_to_flatset(grid):
    return tuple(sorted(n for row in grid for n in row if n is not None))

def generate_unique_tickets(n):
    tickets = []
    seen = set()
    attempts = 0
    while len(tickets) < n:
        attempts += 1
        if attempts > n * 2000:
            raise RuntimeError("Too many attempts to generate unique tickets; reduce N or try again.")
        grid = generate_single_ticket()
        key = ticket_to_flatset(grid)
        if key in seen:
            continue
        seen.add(key)
        ticket_id = f"T{len(tickets)+1:04d}-{int(datetime.utcnow().timestamp())%10000}"
        tickets.append((ticket_id, grid))
    return tickets

# --------------------
# DB Persistence
# --------------------
def save_tickets_to_db(tickets):
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()
    for tid, grid in tickets:
        cur.execute("INSERT OR REPLACE INTO tickets (id, numbers_json, created_at) VALUES (?,?,?)",
                    (tid, json.dumps(grid), now))
    conn.commit()

def load_all_tickets():
    cur = conn.cursor()
    cur.execute("SELECT id, numbers_json FROM tickets ORDER BY id")
    rows = cur.fetchall()
    tickets = []
    for r in rows:
        tickets.append((r[0], json.loads(r[1])))
    return tickets

def load_ticket(ticket_id):
    cur = conn.cursor()
    cur.execute("SELECT id, numbers_json FROM tickets WHERE id=?", (ticket_id,))
    r = cur.fetchone()
    if not r:
        return None
    return (r[0], json.loads(r[1]))

# --------------------
# Game State helpers
# --------------------
def set_game_state(key, value):
    cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO game_state (key, value) VALUES (?,?)", (key, json.dumps(value)))
    conn.commit()

def get_game_state(key, default=None):
    cur = conn.cursor()
    cur.execute("SELECT value FROM game_state WHERE key=?", (key,))
    r = cur.fetchone()
    if not r:
        return default
    return json.loads(r[0])

def reset_game_state(preserve_tickets=False):
    # preserve tickets optionally
    set_game_state("called_numbers", [])
    set_game_state("winners", {
        "early_five": None,
        "top_line": None,
        "middle_line": None,
        "bottom_line": None,
        "full_house": None
    })
    set_game_state("last_call_at", None)
    if not preserve_tickets:
        cur = conn.cursor()
        cur.execute("DELETE FROM claims")
        conn.commit()

# --------------------
# Claims
# --------------------
def submit_claim(ticket_id, claim_type, claimant_name):
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()
    t = load_ticket(ticket_id)
    if not t:
        return None
    _, grid = t
    nums = list(n for row in grid for n in row if n is not None)
    cur.execute("INSERT INTO claims (ticket_id, claim_type, claimant_name, numbers_json, validated, validated_at, created_at) VALUES (?,?,?,?,?,?,?)",
                (ticket_id, claim_type, claimant_name, json.dumps(nums), 0, None, now))
    conn.commit()
    return cur.lastrowid

def validate_claim(claim_id, valid=True):
    cur = conn.cursor()
    cur.execute("UPDATE claims SET validated=?, validated_at=? WHERE id=?", (1 if valid else 0, datetime.utcnow().isoformat(), claim_id))
    conn.commit()

# --------------------
# Utilities: ticket numbers set
# --------------------
def ticket_numbers_set(grid):
    return set(n for row in grid for n in row if n is not None)

# --------------------
# Ticket Image Render (Pillow updated API)
# --------------------
def render_ticket_image(grid, ticket_id):
    img = Image.new("RGB", (TICKET_WIDTH_PX, TICKET_HEIGHT_PX), color=(255,255,255))
    draw = ImageDraw.Draw(img)
    try:
        if FONT_PATH and os.path.exists(FONT_PATH):
            font = ImageFont.truetype(FONT_PATH, 18)
            header_font = ImageFont.truetype(FONT_PATH, 20)
        else:
            font = ImageFont.load_default()
            header_font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
        header_font = ImageFont.load_default()

    draw.text((10,6), f"Ticket {ticket_id}", font=header_font, fill=(0,0,0))

    top = 40
    left = 10
    cell_w = (TICKET_WIDTH_PX - left*2) // 9
    cell_h = (TICKET_HEIGHT_PX - top - 10) // 3

    for r in range(3):
        for c in range(9):
            x0 = left + c*cell_w
            y0 = top + r*cell_h
            x1 = x0 + cell_w
            y1 = y0 + cell_h
            draw.rectangle([x0, y0, x1, y1], outline=(0,0,0), width=1)
            val = grid[r][c]
            if val is not None:
                txt = str(val)
                # pillow >=10: use textbbox
                try:
                    bbox = draw.textbbox((0, 0), txt, font=font)
                    w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
                except AttributeError:
                    # older pillow fallback
                    w, h = draw.textsize(txt, font=font)
                tx = x0 + (cell_w - w)//2
                ty = y0 + (cell_h - h)//2
                draw.text((tx,ty), txt, font=font, fill=(0,0,0))
    b = BytesIO()
    img.save(b, format="PNG")
    b.seek(0)
    return b

# --------------------
# Exports
# --------------------
def tickets_to_dataframe(tickets):
    rows = []
    for tid, grid in tickets:
        nums = [str(n) for row in grid for n in row if n is not None]
        rows.append({"ticket_id": tid, "numbers": ",".join(nums)})
    return pd.DataFrame(rows)

def create_tickets_zip(tickets):
    mem = BytesIO()
    with ZipFile(mem, "w") as zf:
        for tid, grid in tickets:
            img_b = render_ticket_image(grid, tid)
            zf.writestr(f"{tid}.png", img_b.read())
    mem.seek(0)
    return mem

# --------------------
# Claim validation logic (based on called numbers set)
# --------------------
def check_claim_validity(ticket_grid, called_numbers_set, claim_type):
    nums = ticket_numbers_set(ticket_grid)
    if claim_type == "Early 5":
        return len(nums & called_numbers_set) >= 5
    if claim_type == "Top Line":
        top_row = set(n for n in ticket_grid[0] if n is not None)
        return top_row.issubset(called_numbers_set)
    if claim_type == "Middle Line":
        mid_row = set(n for n in ticket_grid[1] if n is not None)
        return mid_row.issubset(called_numbers_set)
    if claim_type == "Bottom Line":
        bottom_row = set(n for n in ticket_grid[2] if n is not None)
        return bottom_row.issubset(called_numbers_set)
    if claim_type == "Full House":
        return nums.issubset(called_numbers_set)
    return False

# --------------------
# Streamlit UI
# --------------------
st.set_page_config(page_title="Tambola (Housie) - Host Dashboard", layout="wide")
st.title("Tambola / Housie — Host & Player App")

# sidebar admin controls
st.sidebar.header("Admin Controls")
mode = st.sidebar.selectbox("Mode", ["Host (Admin)", "Player View / Claim"])

if st.sidebar.button("Initialize DB & Reset Game"):
    reset_game_state(preserve_tickets=True)
    cur = conn.cursor()
    cur.execute("DELETE FROM claims")
    conn.commit()
    st.sidebar.success("Game state reset. Claims cleared. Called numbers cleared.")

# Host mode
if mode == "Host (Admin)":
    st.sidebar.subheader("Ticket generation")
    n = st.sidebar.number_input("Number of tickets to generate", min_value=1, max_value=2000, value=DEFAULT_TICKET_COUNT, step=1)
    if st.sidebar.button("Generate Unique Tickets (append to DB)"):
        with st.spinner(f"Generating {n} unique tickets..."):
            tickets = generate_unique_tickets(n)
            save_tickets_to_db(tickets)
        st.sidebar.success(f"Generated {n} tickets and saved to DB.")
    if st.sidebar.button("Load existing tickets from DB"):
        loaded = load_all_tickets()
        st.sidebar.success(f"Loaded {len(loaded)} tickets.")
    st.sidebar.markdown("---")

    st.sidebar.subheader("Game Controls")
    called = get_game_state("called_numbers", [])
    if called is None:
        called = []
    if st.sidebar.button("Start/Reset Called Numbers (keep tickets)"):
        reset_game_state(preserve_tickets=True)
        st.sidebar.success("Called numbers reset.")
        called = []
    if st.sidebar.button("Clear ALL (tickets + state)"):
        cur = conn.cursor()
        cur.execute("DELETE FROM tickets")
        conn.commit()
        reset_game_state(preserve_tickets=False)
        st.sidebar.success("All tickets and state cleared.")

    st.sidebar.markdown("### Number Caller")
    if st.sidebar.button("Call Random Number"):
        all_called = set(get_game_state("called_numbers", [])) or set()
        remaining = set(range(1,91)) - all_called
        if not remaining:
            st.sidebar.error("All numbers have been called.")
        else:
            nxt = random.choice(list(remaining))
            all_called.add(nxt)
            set_game_state("called_numbers", sorted(list(all_called)))
            set_game_state("last_call_at", datetime.utcnow().isoformat())
            st.sidebar.success(f"Called: {nxt}")
    st.sidebar.markdown("Manually call a number")
    manual = st.sidebar.number_input("Manual call number (1-90)", min_value=1, max_value=90, value=1, step=1)
    if st.sidebar.button("Manually call this number"):
        all_called = set(get_game_state("called_numbers", [])) or set()
        if manual in all_called:
            st.sidebar.warning(f"{manual} already called.")
        else:
            all_called.add(manual)
            set_game_state("called_numbers", sorted(list(all_called)))
            set_game_state("last_call_at", datetime.utcnow().isoformat())
            st.sidebar.success(f"Called: {manual}")

    # Host dashboard area
    st.header("Host Dashboard — Tickets & Exports")
    tickets = load_all_tickets()
    st.write(f"Total tickets in DB: **{len(tickets)}**")
    if len(tickets) == 0:
        st.info("No tickets found. Use the sidebar to generate tickets.")
    else:
        df = tickets_to_dataframe(tickets)
        st.dataframe(df.head(200))

        # CSV
        csv = df.to_csv(index=False).encode()
        st.download_button("Download tickets (CSV)", data=csv, file_name="tambola_tickets.csv", mime="text/csv")

        # Excel (BytesIO + context manager)
        towrite = BytesIO()
        with pd.ExcelWriter(towrite, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="tickets")
        towrite.seek(0)
        st.download_button("Download tickets (Excel)", data=towrite, file_name="tambola_tickets.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # ZIP of PNGs
        if st.button("Generate ZIP of ticket PNGs"):
            with st.spinner("Rendering images..."):
                zip_mem = create_tickets_zip(tickets)
            st.download_button("Download tickets (ZIP PNG)", data=zip_mem, file_name="tickets_images.zip", mime="application/zip")

    st.markdown("---")
    st.header("Live Caller Panel")
    called_numbers = get_game_state("called_numbers", [])
    if called_numbers is None:
        called_numbers = []
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Called Numbers")
        st.write("**Numbers called:** " + (", ".join(map(str,called_numbers)) if called_numbers else "None"))
        if st.button("Show random sample ticket (host view)"):
            tickets = load_all_tickets()
            if tickets:
                sample = random.choice(tickets)
                tid, grid = sample
                st.write(f"Sample: {tid}")
                img_b = render_ticket_image(grid, tid)
                st.image(img_b)
            else:
                st.info("No tickets available.")
    with col2:
        st.subheader("Quick actions")
        st.markdown("### Claims")
        cur = conn.cursor()
        cur.execute("SELECT id, ticket_id, claim_type, claimant_name, numbers_json, validated, validated_at FROM claims ORDER BY id DESC LIMIT 200")
        claims = cur.fetchall()
        if claims:
            claims_df = pd.DataFrame([{"id":c[0],"ticket_id":c[1],"claim_type":c[2],"claimant":c[3],"numbers":json.loads(c[4]) if c[4] else None,"validated":c[5],"validated_at":c[6]} for c in claims])
            st.dataframe(claims_df)
            sel = st.number_input("Claim ID to validate", min_value=0, value=0, step=1)
            if sel > 0 and st.button("Validate Selected Claim"):
                cur.execute("SELECT id, ticket_id, claim_type FROM claims WHERE id=?", (sel,))
                r = cur.fetchone()
                if r:
                    cid, tidd, ctype = r
                    t = load_ticket(tidd)
                    if t:
                        _, grid = t
                        called = set(get_game_state("called_numbers", []))
                        ok = check_claim_validity(grid, called, ctype)
                        validate_claim(cid, valid=ok)
                        if ok:
                            st.success(f"Claim {cid} validated: TRUE")
                        else:
                            st.error(f"Claim {cid} validated: FALSE")
                    else:
                        st.error("Ticket not found.")
                else:
                    st.error("Claim ID not found.")
        else:
            st.write("No claims yet.")

# Player mode
elif mode == "Player View / Claim":
    st.header("Player Portal — View ticket & submit claims")
    st.write("Enter your Ticket ID (from the host) to view ticket and check against called numbers.")
    ticket_id = st.text_input("Ticket ID (case-sensitive)")
    if st.button("Load Ticket"):
        if not ticket_id:
            st.warning("Enter a Ticket ID.")
        else:
            loaded = load_ticket(ticket_id)
            if not loaded:
                st.error("Ticket not found. Check with host.")
            else:
                tid, grid = loaded
                st.subheader(f"Ticket: {tid}")
                called = set(get_game_state("called_numbers", []))
                # render ticket image and highlight called numbers
                img = Image.new("RGB", (TICKET_WIDTH_PX, TICKET_HEIGHT_PX), color=(255,255,255))
                draw = ImageDraw.Draw(img)
                try:
                    if FONT_PATH and os.path.exists(FONT_PATH):
                        font = ImageFont.truetype(FONT_PATH, 18)
                        header_font = ImageFont.truetype(FONT_PATH, 20)
                    else:
                        font = ImageFont.load_default()
                        header_font = ImageFont.load_default()
                except:
                    font = ImageFont.load_default()
                    header_font = ImageFont.load_default()
                draw.text((10,6), f"Ticket {tid}", font=header_font, fill=(0,0,0))
                top = 40; left = 10
                cell_w = (TICKET_WIDTH_PX - left*2) // 9
                cell_h = (TICKET_HEIGHT_PX - top - 10) // 3
                for r in range(3):
                    for c in range(9):
                        x0 = left + c*cell_w
                        y0 = top + r*cell_h
                        x1 = x0 + cell_w
                        y1 = y0 + cell_h
                        draw.rectangle([x0, y0, x1, y1], outline=(0,0,0), width=1)
                        val = grid[r][c]
                        if val is not None:
                            txt = str(val)
                            try:
                                bbox = draw.textbbox((0, 0), txt, font=font)
                                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                            except AttributeError:
                                w, h = draw.textsize(txt, font=font)
                            tx = x0 + (cell_w - w)//2
                            ty = y0 + (cell_h - h)//2
                            if val in called:
                                draw.rectangle([x0+2,y0+2,x1-2,y1-2], fill=(200,255,200))
                            draw.text((tx,ty), txt, font=font, fill=(0,0,0))
                b = BytesIO(); img.save(b, format="PNG"); b.seek(0)
                st.image(b)

                st.write("Numbers called so far: " + (", ".join(map(str, sorted(list(called)))) if called else "None"))

                st.markdown("### Submit a Claim")
                claimant = st.text_input("Your name (for claim record)", key=f"name_{ticket_id}")
                claim_type = st.selectbox("Claim type", ["Early 5","Top Line","Middle Line","Bottom Line","Full House"], key=f"claim_{ticket_id}")
                if st.button("Submit Claim"):
                    if not claimant:
                        st.warning("Enter your name.")
                    else:
                        cid = submit_claim(tid, claim_type, claimant)
                        if cid:
                            st.success(f"Claim submitted. Claim ID: {cid}. Host will validate soon.")
                        else:
                            st.error("Failed to submit claim. Contact host.")
                        # Optional auto-validate (local quick check)
                        if st.checkbox("Try auto-validate now (quick check)"):
                            calledset = set(get_game_state("called_numbers", []))
                            ok = check_claim_validity(grid, calledset, claim_type)
                            # do not auto-update DB validation unless host wants; show result
                            if ok:
                                st.success("Auto-check: VALID (inform host)")
                            else:
                                st.error("Auto-check: INVALID (not enough numbers called)")

# Footer & Winner auto-detection
st.markdown("---")
st.caption("Notes: For production, move DB to Postgres and secure ticket access (OTP/email).")

# Auto-detect winners across tickets (first valid claim/winner)
called_set = set(get_game_state("called_numbers", [])) or set()
winners = get_game_state("winners", {
    "early_five": None,
    "top_line": None,
    "middle_line": None,
    "bottom_line": None,
    "full_house": None
})

# detect winners if not already set
tickets = load_all_tickets()
for tid, grid in tickets:
    # skip if all winners assigned
    if all(v is not None for v in winners.values()):
        break
    nums = ticket_numbers_set(grid)
    # Early five
    if winners["early_five"] is None and len(nums & called_set) >= 5:
        winners["early_five"] = tid
    # top line
    if winners["top_line"] is None:
        top_set = set(n for n in grid[0] if n is not None)
        if top_set.issubset(called_set):
            winners["top_line"] = tid
    # middle line
    if winners["middle_line"] is None:
        mid_set = set(n for n in grid[1] if n is not None)
        if mid_set.issubset(called_set):
            winners["middle_line"] = tid
    # bottom line
    if winners["bottom_line"] is None:
        bot_set = set(n for n in grid[2] if n is not None)
        if bot_set.issubset(called_set):
            winners["bottom_line"] = tid
    # full house
    if winners["full_house"] is None and nums.issubset(called_set):
        winners["full_house"] = tid

# persist winners
set_game_state("winners", winners)

st.header("🏆 Winners (Auto-detected)")
for k, v in winners.items():
    if v:
        st.success(f"{k.replace('_',' ').title()}: {v}")
    else:
        st.info(f"{k.replace('_',' ').title()}: Not yet")

# show recent claims log
st.markdown("### Recent claims (last 50)")
cur = conn.cursor()
cur.execute("SELECT id, ticket_id, claim_type, claimant_name, validated, created_at FROM claims ORDER BY id DESC LIMIT 50")
rows = cur.fetchall()
if rows:
    dfc = pd.DataFrame(rows, columns=["id","ticket_id","claim_type","claimant","validated","created_at"])
    st.dataframe(dfc)
else:
    st.write("No claims yet.")
