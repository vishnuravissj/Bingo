# app.py
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

# ---------- Constants ----------
DB_PATH = "tambola.db"
DEFAULT_PLAYER_COUNT = 200
COLUMN_RANGES = [(1,10),(11,20),(21,30),(31,40),(41,50),(51,60),(61,70),(71,80),(81,90)]
FONT_PATH = None  # use default PIL font; set to ttf path if available for nicer look
TICKET_WIDTH_PX = 540
TICKET_HEIGHT_PX = 300

# ---------- DB helpers ----------
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
                    validated_at TEXT
                )""")
    conn.commit()
    return conn

conn = init_db()

# ---------- Tambola ticket generation ----------
def generate_single_ticket():
    # Generate a single 3x9 ticket following standard tambola constraints:
    # - 3 rows, 9 columns
    # - each row exactly 5 numbers (so 15 numbers total)
    # - column i contains numbers from COLUMN_RANGES[i]
    # - each column must have 1-3 numbers (per ticket)
    # We'll produce a grid with None or int.
    cols = [[] for _ in range(9)]
    # Step 1: Choose how many numbers in each column (each column: 0-3, but total must be 15 and each row must have 5)
    # Simpler standard approach: Fill each column with at least 1 number until 15 numbers are allocated, but ensure max 3.
    # We'll do heuristic: start by assigning 1 to columns until we have at most 15, then add remaining randomly keeping max 3.
    counts = [0]*9
    # Start by giving each column 1 number (but only until we approach 15)
    remaining = 15
    # Initial minimal distribution: each column gets 0; we will ensure 15 numbers with constraints
    # Greedy: give 1 to random columns until remaining==0 or all have 1 and then add extras randomly
    # More robust approach: first give 1 to random columns until we assign at least 9 or less depending on remaining
    # We'll ensure feasibility by starting with an empty array and adding numbers one by one with constraints.
    while remaining > 0:
        idx = random.randrange(9)
        if counts[idx] < 3:
            # prevent impossible row distributions: max 3 per column
            counts[idx] += 1
            remaining -= 1
    # Now we have counts per column summing to 15, each 1..3
    # Next: within each column, pick unique numbers from that column's range
    grid_cols = []
    for i, cnt in enumerate(counts):
        lo, hi = COLUMN_RANGES[i]
        choices = list(range(lo, hi+1))
        selected = random.sample(choices, cnt) if cnt>0 else []
        selected.sort()
        grid_cols.append(selected)
    # Now distribute the column numbers into 3 rows such that each row has exactly 5 numbers
    # Start with empty 3x9 grid (None)
    grid = [[None]*9 for _ in range(3)]
    # For each column, distribute its numbers into random rows (no two numbers in same column in same row)
    for col_idx, numbers in enumerate(grid_cols):
        rows_available = [0,1,2]
        random.shuffle(rows_available)
        for n_idx, num in enumerate(numbers):
            # pick a row that currently has <5 numbers and no number in this column
            placed = False
            # sort rows by current filled count (prefer rows with fewer to balance)
            row_order = sorted(rows_available, key=lambda r: sum(1 for x in grid[r] if x is not None))
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
                # If still not placed (rare), restart generation
                return generate_single_ticket()
    # Final validation: each row must have 5 numbers
    for r in range(3):
        if sum(1 for x in grid[r] if x is not None) != 5:
            return generate_single_ticket()
    return grid

def ticket_to_flatset(grid):
    nums = sorted([n for row in grid for n in row if n is not None])
    return tuple(nums)

def generate_unique_tickets(n):
    # Generate n unique tickets (by set of numbers)
    tickets = []
    seen = set()
    attempts = 0
    while len(tickets) < n:
        attempts += 1
        if attempts > n * 1000:
            raise RuntimeError("Too many attempts to generate unique tickets; reduce N or try again.")
        grid = generate_single_ticket()
        key = ticket_to_flatset(grid)
        if key in seen:
            continue
        seen.add(key)
        ticket_id = f"T{len(tickets)+1:04d}-{int(datetime.utcnow().timestamp())%10000}"
        tickets.append((ticket_id, grid))
    return tickets

# ---------- Persistence ----------
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

# ---------- Game state ----------
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

def reset_game_state():
    set_game_state("called_numbers", [])
    set_game_state("started_at", None)
    set_game_state("last_call_at", None)

# ---------- Claim logic ----------
def submit_claim(ticket_id, claim_type, claimant_name, numbers):
    cur = conn.cursor()
    cur.execute("INSERT INTO claims (ticket_id, claim_type, claimant_name, numbers_json, validated, validated_at) VALUES (?,?,?,?,?,?)",
                (ticket_id, claim_type, claimant_name, json.dumps(numbers), 0, None))
    conn.commit()
    return cur.lastrowid

def validate_claim(claim_id, valid=True):
    cur = conn.cursor()
    cur.execute("UPDATE claims SET validated=?, validated_at=? WHERE id=?", (1 if valid else 0, datetime.utcnow().isoformat(), claim_id))
    conn.commit()

# ---------- Utility: flatten ticket numbers ----------
def ticket_numbers_set(grid):
    return set(n for row in grid for n in row if n is not None)

# ---------- Drawing ticket image ----------
def render_ticket_image(grid, ticket_id):
    # Create a PNG for the 3x9 grid
    img = Image.new("RGB", (TICKET_WIDTH_PX, TICKET_HEIGHT_PX), color=(255,255,255))
    draw = ImageDraw.Draw(img)
    # load font
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

    # draw header
    draw.text((10,6), f"Ticket {ticket_id}", font=header_font, fill=(0,0,0))

    # grid params
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
                w,h = draw.textsize(txt, font=font)
                tx = x0 + (cell_w - w)//2
                ty = y0 + (cell_h - h)//2
                draw.text((tx,ty), txt, font=font, fill=(0,0,0))
    # return bytes
    b = BytesIO()
    img.save(b, format="PNG")
    b.seek(0)
    return b

# ---------- Exports ----------
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

# ---------- Claim validation rules ----------
def check_claim_validity(ticket_grid, called_numbers_set, claim_type):
    nums = ticket_numbers_set(ticket_grid)
    # Early Five = any 5 numbers marked earliest — but to validate, require that at least 5 numbers of ticket are in called numbers.
    if claim_type == "Early 5":
        return len(nums & called_numbers_set) >= 5
    if claim_type == "Top Line":
        # top row all numbers
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
    # unknown claim
    return False

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Tambola (Housie) - Host Dashboard", layout="wide")
st.title("Tambola / Housie — Host & Player App")
st.caption("Build: standalone Streamlit app — generates tickets, calls numbers, validates claims. (Designed for ~200 players)")

# Sidebar: Admin controls
st.sidebar.header("Admin Controls")
mode = st.sidebar.selectbox("Mode", ["Host (Admin)", "Player View / Claim"])

if st.sidebar.button("Initialize DB & Reset Game"):
    # keep tickets but reset called numbers and claims
    reset_game_state()
    cur = conn.cursor()
    cur.execute("DELETE FROM claims")
    conn.commit()
    st.sidebar.success("Game state reset. Claims cleared. Called numbers cleared.")

if mode == "Host (Admin)":
    st.sidebar.subheader("Ticket generation")
    n = st.sidebar.number_input("Number of tickets to generate", min_value=1, max_value=2000, value=DEFAULT_PLAYER_COUNT, step=1)
    if st.sidebar.button("Generate Unique Tickets (will append to DB)"):
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
    if st.sidebar.button("Start/Reset Called Numbers"):
        reset_game_state()
        st.sidebar.success("Called numbers reset.")
        called = []
    # draw number
    st.sidebar.markdown("### Number Caller")
    if st.sidebar.button("Call Random Number"):
        all_called = set(get_game_state("called_numbers", []))
        remaining = set(range(1,91)) - all_called
        if not remaining:
            st.sidebar.error("All numbers have been called.")
        else:
            nxt = random.choice(list(remaining))
            all_called.add(nxt)
            set_game_state("called_numbers", sorted(list(all_called)))
            set_game_state("last_call_at", datetime.utcnow().isoformat())
            called = sorted(list(all_called))
            st.sidebar.success(f"Called: {nxt}")
    st.sidebar.markdown("Called numbers so far: " + (", ".join(map(str,called)) if called else "None"))
    st.sidebar.markdown("---")

    st.header("Host Dashboard — Tickets & Exports")
    tickets = load_all_tickets()
    st.write(f"Total tickets in DB: **{len(tickets)}**")
    if len(tickets) == 0:
        st.info("No tickets found. Use the sidebar to generate tickets.")
    else:
        df = tickets_to_dataframe(tickets)
        st.dataframe(df.head(200))

        # Export CSV / Excel
        csv = df.to_csv(index=False).encode()
        st.download_button("Download tickets (CSV)", data=csv, file_name="tambola_tickets.csv", mime="text/csv")

        # Excel
        towrite = BytesIO()
        with pd.ExcelWriter(towrite, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="tickets")
            #writer.save()
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
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Called Numbers")
        called_numbers = st.session_state.get("called_numbers_cache", called_numbers)
        # Refresh from DB
        called_numbers = get_game_state("called_numbers", [])
        grid_display = np.zeros((10,9), dtype=int)  # 1-based numbers shown in a grid for UX? We'll just list
        if called_numbers:
            st.write("**Numbers called:** " + ", ".join(map(str, called_numbers)))
        else:
            st.write("No numbers called yet.")
        # Quick manual call
        manual = st.number_input("Manual call number (1-90)", min_value=1, max_value=90, value=1, step=1)
        if st.button("Manually call this number"):
            all_called = set(get_game_state("called_numbers", []))
            if manual in all_called:
                st.warning(f"{manual} already called.")
            else:
                all_called.add(manual)
                set_game_state("called_numbers", sorted(list(all_called)))
                st.success(f"Called: {manual}")
    with col2:
        st.subheader("Quick actions")
        if st.button("Show random sample ticket (host view)"):
            tickets = load_all_tickets()
            if tickets:
                sample = random.choice(tickets)
                tid, grid = sample
                st.write(f"Sample: {tid}")
                # render inline
                img_b = render_ticket_image(grid, tid)
                st.image(img_b)
            else:
                st.info("No tickets available.")
        # Show claims
        st.markdown("### Claims")
        cur = conn.cursor()
        cur.execute("SELECT id, ticket_id, claim_type, claimant_name, numbers_json, validated, validated_at FROM claims ORDER BY id DESC LIMIT 200")
        claims = cur.fetchall()
        if claims:
            claims_df = pd.DataFrame([{"id":c[0],"ticket_id":c[1],"claim_type":c[2],"claimant":c[3],"numbers":json.loads(c[4]) if c[4] else None,"validated":c[5],"validated_at":c[6]} for c in claims])
            st.dataframe(claims_df)
            # Validate selected claim
            sel = st.number_input("Claim ID to validate", min_value=0, value=0, step=1)
            if sel > 0:
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
                # render ticket image and also annotate cells that are called
                # We'll render a PIL image but highlight called numbers
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
                            #w,h = draw.textsize(txt, font=font)
                            bbox = draw.textbbox((0, 0), txt, font=font)
                            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                            tx = x0 + (cell_w - w)//2
                            ty = y0 + (cell_h - h)//2
                            if val in called:
                                # highlight
                                draw.rectangle([x0+2,y0+2,x1-2,y1-2], fill=(200,255,200))
                            draw.text((tx,ty), txt, font=font, fill=(0,0,0))
                b = BytesIO(); img.save(b, format="PNG"); b.seek(0)
                st.image(b)

                st.write("Numbers called so far: " + (", ".join(map(str, sorted(list(called)))) if called else "None"))

                st.markdown("### Submit a Claim")
                claimant = st.text_input("Your name (for claim record)")
                claim_type = st.selectbox("Claim type", ["Early 5","Top Line","Middle Line","Bottom Line","Full House"])
                if st.button("Submit Claim"):
                    if not claimant:
                        st.warning("Enter your name.")
                    else:
                        # record claim
                        nums = list(ticket_numbers_set(grid))
                        cid = submit_claim(tid, claim_type, claimant, nums)
                        st.success(f"Claim submitted. Claim ID: {cid}. Host will validate soon.")
                        # Optionally auto-validate
                        if st.checkbox("Try auto-validate now (host override)"):
                            calledset = set(get_game_state("called_numbers", []))
                            ok = check_claim_validity(grid, calledset, claim_type)
                            validate_claim(cid, valid=ok)
                            if ok:
                                st.success("Auto-validated: TRUE — please inform host.")
                            else:
                                st.error("Auto-validated: FALSE — not enough numbers called yet.")

# Footer
st.markdown("---")
st.caption("App built for scale; for production use consider hosting on Streamlit Cloud and using an external DB (Postgres) if you expect many concurrent requests.")
