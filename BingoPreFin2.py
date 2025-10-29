# app.py
"""
Tambola — Corporate Blue & White (section-wise styling)
Full single-file Streamlit app:
- Admin-key protected admin panel (enter key in sidebar)
- Section-wise styling (cards per section) using CSS
- Ticket generation, claims, number draw, exports (CSV/ZIP)
- Prizes: Early 5, 4 Corners, Top/Middle/Bottom Line, Full House
- Player ticket load persistence, submit claims, admin approve/reject
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
st.set_page_config(page_title="Tambola — Blue & White", layout="wide")
ADMIN_KEY = "VISHNU2025"  # change this before production
DEFAULT_TICKET_COUNT = 200
TICKET_ROWS, TICKET_COLS = 3, 9
TICKET_IMG_W, TICKET_IMG_H = 540, 300
FONT_PATH = None  # optional .ttf path

# -------------------------
# SECTION-STYLES (section-wise)
# We'll add small CSS classes for card-like containers applied per section.
# -------------------------
st.markdown(
    """
    <style>
    /* Basic color tokens */
    :root {
        --bg: #f8fbff;
        --card-bg: #ffffff;
        --muted: #475569;
        --primary: #0078D4; /* blue */
        --accent: #0b5fb0;
        --border: #e6eef8;
    }

    /* Page background */
    .stApp {
        background: var(--bg);
        color: var(--muted);
        font-family: "Inter", "Segoe UI", Roboto, Arial, sans-serif;
    }

    /* Generic card */
    .card {
        background: var(--card-bg);
        border-radius: 10px;
        padding: 14px;
        box-shadow: 0 4px 10px rgba(16,24,40,0.04);
        border: 1px solid var(--border);
        margin-bottom: 12px;
    }

    /* Header card */
    .header-card {
        display:flex; align-items:center; justify-content:space-between;
        gap:12px;
    }
    .title {
        font-size: 20px; font-weight:700; color: var(--primary);
    }
    .subtitle { color: var(--muted); font-size:13px; }

    /* Number board */
    .num-board { display:grid; grid-template-columns: repeat(9, 1fr); gap:8px; margin-top:8px; }
    .num-tile {
        background: #f1f7ff;
        border-radius: 8px;
        padding: 8px;
        text-align: center;
        font-weight:700;
        color: var(--primary);
        border: 1px solid var(--border);
    }
    .num-tile.called {
        background: linear-gradient(180deg, #e6f2ff, #d7ecff);
        color: #033a6b;
        box-shadow: 0 6px 20px rgba(3,58,107,0.06);
        transform: translateY(-3px);
    }
    .num-tile.last {
        border: 1px solid #004a99;
        box-shadow: 0 8px 30px rgba(0,72,164,0.12);
    }

    /* Ticket html */
    .ticket {
        background: linear-gradient(180deg,#ffffff,#fbfdff);
        border-radius:8px; padding:10px; border:1px solid var(--border);
    }
    .ticket .row { display:flex; gap:6px; margin-bottom:6px; }
    .ticket .cell {
        width:44px; height:36px; border-radius:6px;
        display:flex; align-items:center; justify-content:center;
        font-weight:700; color:var(--primary);
        background: #f7fbff; border:1px solid #e6eef8;
    }
    .ticket .cell.called {
        background: #dff3ff; color: #012a3f;
    }

    /* small helpers */
    .muted { color: var(--muted); font-size:13px; }
    .section-title { color: var(--primary); font-weight:700; margin-bottom:8px; }
    .small { font-size:12px; color: #60748a; }

    /* make streamlit widgets more visible */
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        border-radius:8px !important;
        border:1px solid #dbeaf8 !important;
        padding:8px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# SESSION STATE INIT
# -------------------------
st.session_state.setdefault("tickets", {})                # Dict[str, List[List[int]]]
st.session_state.setdefault("called_numbers", [])         # List[int]
st.session_state.setdefault("claims_queue", [])           # List[dict]
st.session_state.setdefault("claimed_prizes", {})         # Dict[str, str]
st.session_state.setdefault("loaded_ticket_id", None)
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

def generate_single_ticket():
    while True:
        counts = [0]*9
        rem = 15
        while rem>0:
            idx = random.randrange(9)
            if counts[idx] < 3:
                counts[idx]+=1; rem-=1
        cols = []
        valid = True
        for i,cnt in enumerate(counts):
            lo,hi = COLUMN_RANGES[i]
            pool = list(range(lo,hi+1))
            if cnt > len(pool):
                valid = False; break
            cols.append(sorted(random.sample(pool,cnt)) if cnt>0 else [])
        if not valid:
            continue
        grid = [[None]*9 for _ in range(3)]
        ok = True
        for cidx, nums in enumerate(cols):
            rows = [0,1,2]; random.shuffle(rows)
            for n in nums:
                placed=False
                rows_sorted = sorted(rows, key=lambda r: sum(1 for x in grid[r] if x is not None))
                for r in rows_sorted:
                    if grid[r][cidx] is None and sum(1 for x in grid[r] if x is not None) < 5:
                        grid[r][cidx]=n; placed=True; break
                if not placed:
                    for r in range(3):
                        if grid[r][cidx] is None and sum(1 for x in grid[r] if x is not None) < 5:
                            grid[r][cidx]=n; placed=True; break
                if not placed:
                    ok=False; break
            if not ok: break
        if ok and all(sum(1 for x in grid[r] if x is not None)==5 for r in range(3)):
            return [[(cell if cell is not None else 0) for cell in row] for row in grid]

def gen_unique_tickets(n:int, prefix:str="T"):
    tickets={}
    seen=set()
    attempts=0
    while len(tickets) < n:
        attempts+=1
        if attempts > n*2000:
            raise RuntimeError("Too many attempts")
        grid = generate_single_ticket()
        key = tuple(sorted(x for row in grid for x in row if x!=0))
        if key in seen: continue
        seen.add(key)
        tid = f"{prefix}{len(tickets)+1:04d}"
        tickets[tid] = grid
    return tickets

# -------------------------
# RENDER TICKET HTML (styled per-section)
# -------------------------
def ticket_html(grid, ticket_id, called:set):
    rows_html = ""
    for r in range(TICKET_ROWS):
        cells = ""
        for c in range(TICKET_COLS):
            val = grid[r][c]
            if val == 0:
                txt = "&nbsp;"
                cls = ""
            else:
                txt = str(val)
                cls = "called" if val in called else ""
            cells += f'<div class="cell {cls}">{txt}</div>'
        rows_html += f'<div class="row">{cells}</div>'
    html = f'''
    <div class="ticket card" style="max-width:420px;">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
         <div style="font-weight:800; color: #0078D4;">{ticket_id}</div>
         <div class="small muted">Called: {", ".join(map(str,sorted(called))) if called else "None"}</div>
      </div>
      {rows_html}
    </div>
    '''
    return html

# -------------------------
# IMAGE RENDER / EXPORT HELPERS
# -------------------------
def render_ticket_image(grid, ticket_id):
    img = Image.new("RGB", (TICKET_IMG_W, TICKET_IMG_H), color=(255,255,255))
    draw = ImageDraw.Draw(img)
    try:
        if FONT_PATH:
            header_font = ImageFont.truetype(FONT_PATH, 18)
            cell_font = ImageFont.truetype(FONT_PATH, 16)
        else:
            header_font = ImageFont.load_default(); cell_font = ImageFont.load_default()
    except Exception:
        header_font = ImageFont.load_default(); cell_font = ImageFont.load_default()

    draw.text((10,6), f"Ticket {ticket_id}", font=header_font, fill=(0,0,0))
    left, top = 10, 40
    cw = (TICKET_IMG_W - left*2)//TICKET_COLS
    ch = (TICKET_IMG_H - top - 10)//TICKET_ROWS
    for r in range(TICKET_ROWS):
        for c in range(TICKET_COLS):
            x0 = left + c*cw; y0 = top + r*ch; x1 = x0+cw; y1 = y0+ch
            draw.rectangle([x0,y0,x1,y1], outline=(200,200,200), width=1)
            v = grid[r][c]
            if v and v!=0:
                txt = str(v)
                try:
                    bbox = draw.textbbox((0,0), txt, font=cell_font)
                    w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
                except AttributeError:
                    w, h = draw.textsize(txt, font=cell_font)
                draw.text((x0 + (cw-w)/2, y0 + (ch-h)/2), txt, font=cell_font, fill=(0,0,0))
    buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0); return buf

def tickets_to_dataframe(tickets:Dict[str,List[List[int]]]):
    rows=[]
    for tid, grid in tickets.items():
        nums = [str(n) for row in grid for n in row if n!=0]
        rows.append({"ticket_id":tid, "numbers": ",".join(nums)})
    return pd.DataFrame(rows)

def create_tickets_zip(tickets:Dict[str,List[List[int]]]):
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w") as zf:
        for tid, grid in tickets.items():
            b = render_ticket_image(grid, tid)
            zf.writestr(f"{tid}.png", b.read())
    mem.seek(0); return mem

# -------------------------
# PRIZE LOGIC
# -------------------------
def ticket_number_set(grid): return set(n for row in grid for n in row if n!=0)

def check_prize_for_ticket(grid, called_set:set, prize_name:str):
    if prize_name == "Early 5":
        return len(ticket_number_set(grid) & called_set) >= 5
    if prize_name == "Top Line":
        top = set(n for n in grid[0] if n!=0); return top.issubset(called_set)
    if prize_name == "Middle Line":
        mid = set(n for n in grid[1] if n!=0); return mid.issubset(called_set)
    if prize_name == "Bottom Line":
        bot = set(n for n in grid[2] if n!=0); return bot.issubset(called_set)
    if prize_name == "4 Corners":
        corners = [grid[0][0], grid[0][8], grid[2][0], grid[2][8]]
        return all((c!=0 and c in called_set) for c in corners)
    if "Full House" in prize_name:
        return ticket_number_set(grid).issubset(called_set)
    return False

# -------------------------
# CLAIMS
# -------------------------
def submit_claim(ticket_id:str, claimant:str, claim_type:str):
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

def validate_claim(cid:int, approver:str, approve:bool):
    for rec in st.session_state.claims_queue:
        if rec["id"] == cid:
            rec["status"] = "approved" if approve else "rejected"
            rec["validated_by"] = approver
            rec["validated_at"] = datetime.utcnow().isoformat()
            if approve and st.session_state.prize_config.get(rec["claim_type"], False):
                if rec["claim_type"] not in st.session_state.claimed_prizes or not st.session_state.claimed_prizes[rec["claim_type"]]:
                    st.session_state.claimed_prizes[rec["claim_type"]] = rec["ticket_id"]
            return True
    return False

# -------------------------
# Number drawing helpers
# -------------------------
def call_random_number():
    rem = set(range(1,91)) - set(st.session_state.called_numbers)
    if not rem: return None
    n = random.choice(list(rem)); st.session_state.called_numbers.append(n); st.session_state.last_draw = n; return n

def call_manual_number(n:int):
    if n in st.session_state.called_numbers: return False
    st.session_state.called_numbers.append(n); st.session_state.last_draw = n; return True

# -------------------------
# SIDEBAR: Admin unlock (option A)
# -------------------------
with st.sidebar:
    st.markdown('<div class="card"><div style="display:flex;align-items:center;gap:12px;"><div style="font-weight:800;color:#0078D4">Tambola — Admin</div><div class="small muted">Host controls</div></div></div>', unsafe_allow_html=True)
    key = st.text_input("Admin Key", type="password")
    if st.button("Unlock Admin"):
        if key == ADMIN_KEY:
            st.session_state.admin_unlocked = True; st.success("Admin unlocked ✅")
        else:
            st.error("Invalid admin key")

    # if admin unlocked show a compact controls block
    if st.session_state.admin_unlocked:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Prizes")
        if "4 Corners" not in st.session_state.prize_config:
            st.session_state.prize_config["4 Corners"] = True
        for pname in list(st.session_state.prize_config.keys()):
            st.session_state.prize_config[pname] = st.checkbox(pname, value=st.session_state.prize_config[pname], key=f"chk_{pname}")
        st.markdown("---")
        st.subheader("Tickets / Export")
        gen_n = st.number_input("Generate count", min_value=1, max_value=2000, value=DEFAULT_TICKET_COUNT, key="gen_n")
        prefix = st.text_input("Prefix", value=st.session_state.ticket_prefix, key="prefix")
        if st.button("Generate Tickets (overwrite)"):
            st.session_state.tickets = gen_unique_tickets(gen_n, prefix)
            st.session_state.called_numbers = []; st.session_state.claimed_prizes = {}; st.session_state.claims_queue = []
            st.success(f"Generated {len(st.session_state.tickets)} tickets")
        if st.button("Export tickets CSV"):
            if not st.session_state.tickets: st.warning("No tickets")
            else:
                df = tickets_to_dataframe(st.session_state.tickets); csv = df.to_csv(index=False).encode(); st.download_button("Download CSV", csv, "tickets.csv", "text/csv")
        if st.button("Export PNGs (ZIP)"):
            if not st.session_state.tickets: st.warning("No tickets")
            else:
                z = create_tickets_zip(st.session_state.tickets); st.download_button("Download ZIP", z, "tickets_images.zip", "application/zip")
        st.markdown("---")
        st.subheader("Number Caller")
        if st.button("Call Random Number"):
            n = call_random_number()
            if n is None: st.warning("All numbers called")
            else: st.success(f"Called {n}")
        manual = st.number_input("Manual call (1-90)", min_value=1, max_value=90, value=1)
        if st.button("Call Manual Number"):
            ok = call_manual_number(manual)
            if not ok: st.warning(f"{manual} already called") 
            else: st.success(f"Called {manual}")
        if st.button("Reset Called Numbers (keep tickets)"):
            st.session_state.called_numbers = []; st.success("Called numbers cleared")
        st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# HEADER (section card)
# -------------------------
st.markdown('<div class="card header-card"><div><div class="title">🎲 Tambola</div><div class="subtitle">Corporate Blue & White — Host-ready</div></div><div class="small muted">Logged in as Admin' + (' ✅' if st.session_state.admin_unlocked else '') + '</div></div>', unsafe_allow_html=True)

# -------------------------
# Number board (section card)
# -------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Number Board</div>', unsafe_allow_html=True)
called = set(st.session_state.called_numbers)
last = st.session_state.last_draw
nums_html = '<div class="num-board">'
# render all 1..90 tiles (section-wise styled)
for n in range(1,91):
    cls = "num-tile"
    if n in called: cls += " called"
    if last and n == last: cls += " last"
    nums_html += f'<div class="{cls}">{n}</div>'
nums_html += '</div>'
st.markdown(nums_html, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Auto-scan eligible (mark prize as eligible placeholder)
# -------------------------
called_set = set(st.session_state.called_numbers)
for prize, enabled in st.session_state.prize_config.items():
    if not enabled: continue
    if prize in st.session_state.claimed_prizes and st.session_state.claimed_prizes[prize]:
        continue
    for tid, grid in st.session_state.tickets.items():
        if check_prize_for_ticket(grid, called_set, prize):
            st.session_state.claimed_prizes.setdefault(prize, None)
            break

st.markdown('---')

# -------------------------
# Player Card (section card)
# -------------------------
col_left, col_right = st.columns([2,1])
with col_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Player Portal — Ticket & Claim</div>', unsafe_allow_html=True)
    pid = st.text_input("Enter your Ticket ID (e.g., T001)", value=st.session_state.get("loaded_ticket_id") or "")
    if st.button("Load Ticket"):
        if not pid:
            st.warning("Enter Ticket ID")
        else:
            if pid not in st.session_state.tickets:
                st.session_state.tickets[pid] = generate_single_ticket()
            st.session_state.loaded_ticket_id = pid; st.success(f"Loaded {pid}")
    if st.session_state.get("loaded_ticket_id"):
        tid = st.session_state.loaded_ticket_id
        if tid in st.session_state.tickets:
            grid = st.session_state.tickets[tid]
            html = ticket_html(grid, tid, called)
            st.markdown(html, unsafe_allow_html=True)
            st.write("Numbers called so far:", sorted(list(called)))
            st.markdown("### Submit Claim")
            claimant = st.text_input("Your name", key=f"name_{tid}")
            active_prizes = [p for p,en in st.session_state.prize_config.items() if en]
            if not active_prizes:
                st.info("No active prizes configured")
            else:
                claim_type = st.selectbox("Select prize", active_prizes, key=f"sel_{tid}")
                if st.button("Submit Claim", key=f"submit_{tid}"):
                    if not claimant:
                        st.warning("Enter your name")
                    else:
                        qualifies = check_prize_for_ticket(grid, called_set, claim_type)
                        cid = submit_claim(tid, claimant, claim_type)
                        st.success(f"Claim submitted (ID {cid}). Admin will validate.")
                        if qualifies:
                            st.info("Quick check: claim appears VALID (admin verification required).")
                        else:
                            st.info("Quick check: claim appears INVALID (admin verification required).")
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Game Snapshot</div>', unsafe_allow_html=True)
    st.write("Numbers called:", len(st.session_state.called_numbers))
    st.write("Last draw:", st.session_state.last_draw if st.session_state.last_draw else "—")
    st.write("Active prizes:", ", ".join([p for p,v in st.session_state.prize_config.items() if v]))
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('---')

# -------------------------
# Claims Queue & Admin Review (section)
# -------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Claims Queue & Admin Review</div>', unsafe_allow_html=True)
if st.session_state.claims_queue:
    df = pd.DataFrame(st.session_state.claims_queue)
    st.dataframe(df[["id","ticket_id","claimant","claim_type","time","status"]])
else:
    st.info("No claims submitted yet.")

if st.session_state.admin_unlocked:
    st.markdown("#### Process a Claim")
    cid = st.number_input("Claim ID", min_value=0, step=1)
    action = st.selectbox("Action", ["Approve","Reject"])
    approver = st.text_input("Approver name", value="Admin", key="approver_name")
    if st.button("Process Claim"):
        if cid <= 0:
            st.warning("Enter a valid Claim ID")
        else:
            recs = [r for r in st.session_state.claims_queue if r["id"]==cid]
            if not recs:
                st.error("Claim not found")
            else:
                rec = recs[0]
                valid_now = check_prize_for_ticket(st.session_state.tickets.get(rec["ticket_id"]), called_set, rec["claim_type"])
                approve = action == "Approve"
                if approve and not valid_now:
                    st.warning("Claim not valid under current board. Approving will force award.")
                validate_claim(cid, approver, approve)
                if approve:
                    st.success(f"Claim {cid} approved; awarded {rec['claim_type']} to {rec['ticket_id']}.")
                else:
                    st.info(f"Claim {cid} rejected.")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('---')

# -------------------------
# Awarded & Eligible Prizes (admin friendly) - section-wise card
# -------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Awarded & Eligible Prizes</div>', unsafe_allow_html=True)
for prize, enabled in st.session_state.prize_config.items():
    if not enabled: continue
    awarded_to = st.session_state.claimed_prizes.get(prize)
    qualifying = [tid for tid, grid in st.session_state.tickets.items() if check_prize_for_ticket(grid, called_set, prize)]
    st.markdown(f"**{prize}**")
    if awarded_to:
        st.success(f"Awarded to: **{awarded_to}**")
    else:
        if qualifying:
            st.info(f"Eligible ({len(qualifying)}): " + ", ".join(qualifying[:30]) + ("" if len(qualifying)<=30 else " ..."))
            # admin quick-award control
            if st.session_state.admin_unlocked:
                pick = st.selectbox(f"Award {prize} to", options=[""] + qualifying, key=f"pick_{prize}")
                if st.button(f"Award {prize}", key=f"awardbtn_{prize}"):
                    if pick:
                        st.session_state.claimed_prizes[prize] = pick
                        st.success(f"Awarded {prize} to {pick}")
                    else:
                        st.warning("Select a ticket first")
        else:
            st.write("No qualifying tickets yet.")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('---')

# -------------------------
# Tickets preview & export (section)
# -------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Tickets Preview & Export</div>', unsafe_allow_html=True)
if st.session_state.tickets:
    df = tickets_to_dataframe(st.session_state.tickets)
    st.dataframe(df.head(500))
    csv = df.to_csv(index=False).encode()
    st.download_button("Download Tickets CSV", csv, "tickets.csv", "text/csv")
    if st.button("Download All Ticket PNGs (ZIP)"):
        z = create_tickets_zip(st.session_state.tickets)
        st.download_button("Download ZIP", z, "tickets_images.zip", "application/zip")
else:
    st.write("No tickets generated yet.")
st.markdown('</div>', unsafe_allow_html=True)

st.caption("If you'd like DB persistence (SQLite) to survive restarts, I can add it next.")
