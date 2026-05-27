import streamlit as st
import json
import os
import io
import base64
import httpx
import subprocess
import re
import tempfile
import chromadb
from collections import defaultdict
from itertools import count
from pathlib import Path

st.set_page_config(page_title="DAYA", page_icon="◈", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@700;800&display=swap');
* { font-family: 'DM Mono', monospace !important; }
body, .stApp { background: #0a0a0f !important; color: #e8e6f0 !important; }
#MainMenu, footer, header, .stDeployButton { display: none !important; }
[data-testid="stSidebar"] { background: #111118 !important; border-right: 1px solid #1e1e2e !important; }
.stTextInput input { background: #111118 !important; border: 1px solid #1e1e2e !important; color: #e8e6f0 !important; border-radius: 6px !important; }
.stTextInput input:focus { border-color: #7c6af7 !important; }
.stButton > button { background: #7c6af7 !important; color: #fff !important; border: none !important; border-radius: 6px !important; }
[data-testid="stFileUploader"] section { background: #0d0d15 !important; border: 1px dashed #2e2e45 !important; border-radius: 8px !important; }
[data-testid="stFileUploaderDropInstructions"], [data-testid="stFileUploader"] section small { display: none !important; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 style="font-family:Syne,sans-serif;font-weight:800;color:#7c6af7;letter-spacing:-0.04em;margin-bottom:0">◈ DAYA</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#6b6880;font-size:0.75rem;letter-spacing:0.15em;text-transform:uppercase">Document Intelligence RAG Pipeline</p>', unsafe_allow_html=True)
st.divider()

# ── Sidebar
with st.sidebar:
    st.markdown("### ⚙ Config")
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    st.divider()
    st.markdown("### 📄 Index Document")
    uploaded = st.file_uploader("Upload", type=["pdf", "pptx", "ppt", "docx"])
    do_index = st.button("⬡ Index") if uploaded and api_key else None
    st.divider()
    perm_dir = Path.home() / ".daya"
    trees = sorted(perm_dir.glob("*_tree.json"), key=os.path.getmtime, reverse=True) if perm_dir.exists() else []
    if trees:
        st.markdown('<span style="color:#6af7a8;font-size:0.75rem">◈ DB ready</span>', unsafe_allow_html=True)
        st.caption(f"↳ {trees[0].name}")
    else:
        st.markdown('<span style="color:#6b6880;font-size:0.75rem">○ No index yet</span>', unsafe_allow_html=True)
    st.divider()
    st.caption("gpt-4o · text-embedding-3-small · ChromaDB")

# ── Stop if no API key
if not api_key:
    st.info("Enter your OpenAI API key in the sidebar to begin.")
    st.stop()

# ── Init clients (only after key is provided)
from openai import OpenAI
client = OpenAI(api_key=api_key)
VISION_MODEL = "gpt-4o"
EMBED_MODEL  = "text-embedding-3-small"
CHUNK_SIZE   = 512
CHUNK_OVERLAP = 16

perm_dir = Path.home() / ".daya"
perm_dir.mkdir(parents=True, exist_ok=True)
chroma_path = perm_dir / "chroma"
chroma_path.mkdir(parents=True, exist_ok=True)
db = chromadb.PersistentClient(path=str(chroma_path))
collection = db.get_or_create_collection(name="daya_demo")

# ── Helpers
def get_embeddings(texts):
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

def chunk_text(text):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks if chunks else [text]

def get_nodes(result):
    dist      = result.get("distances", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    top_5     = [(m["curr_index"], m["next_index"]) for m in metadatas[:5]]
    filtered  = [(d, m) for d, m in zip(dist, metadatas) if d <= 0.85]
    if not filtered: return [], [], []
    freq_map = defaultdict(lambda: {"all_freq": 0})
    for _, m in filtered:
        freq_map[(m["curr_index"], m["next_index"])]["all_freq"] += 1
    valid_ranges = {k for k, v in freq_map.items() if v["all_freq"] > 1 or k in top_5}
    seen, ret_nodes = set(), []
    for _, m in filtered:
        key = (m["curr_index"], m["next_index"])
        if key in valid_ranges and m["node_id"] not in seen:
            seen.add(m["node_id"]); ret_nodes.append(m["node_id"])
    ret_pages   = list(dict.fromkeys(m["curr_index"] for _, m in filtered if m["node_id"] in seen))
    ret_display = list(dict.fromkeys(
        page for _, m in filtered if m["node_id"] in seen
        for page in range(int(m["curr_index"]), int(m["next_index"]) + 1)
    ))
    return ret_nodes, ret_pages, ret_display

def ext_text(filepath, page_list):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    def flatten(nodes):
        flat = []
        for n in nodes:
            flat.append(n)
            if 'children' in n: flat.extend(flatten(n['children']))
        return flat
    all_nodes = flatten(data.get('nodes', []))
    result = []
    for page_num in page_list:
        content = [n.get('text', '') for n in all_nodes if n.get('page_index') == page_num]
        result.append("\n".join(filter(None, content)).strip())
    return result

# ── Indexing
if do_index and uploaded:
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    log_box = st.empty()
    logs = []
    def log(msg):
        logs.append(msg)
        log_box.markdown("\n\n".join(f"`{l}`" for l in logs[-15:]))

    log(f"▶ {uploaded.name}")

    from docling_core.types.doc.document import PictureItem, PictureDescriptionData
    from docling.datamodel.base_models import InputFormat, ItemAndImageEnrichmentElement
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.models.base_model import BaseItemAndImageEnrichmentModel
    from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, LayoutOptions
    from docling_core.types.doc import SectionHeaderItem, TextItem
    from docling_core.types.doc.labels import DocItemLabel
    from docling_core.types.doc.document import TableItem

    SYS_PROMPT = "Extract every visible piece of information from this image as structured Markdown. Always end with a Summary section."

    class VLMOpts(PdfPipelineOptions):
        do_vlm_enrichment: bool = True

    class VLMEnricher(BaseItemAndImageEnrichmentModel):
        images_scale = 2.5
        def __init__(self, enabled):
            self.enabled = enabled
        def is_processable(self, doc, element):
            return self.enabled and isinstance(element, PictureItem)
        def __call__(self, doc, element_batch):
            if not self.enabled: return
            for ee in element_batch:
                element = ee.item
                buf = io.BytesIO(); ee.image.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode()
                completion = client.chat.completions.create(
                    model=VISION_MODEL,
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": SYS_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                    ]}],
                    temperature=0.3, max_tokens=1024,
                )
                element.annotations.append(PictureDescriptionData(
                    text=completion.choices[0].message.content, provenance="vlm"))
                log(f"  ✓ Figure annotated")
                yield element

    class VLMPipeline(StandardPdfPipeline):
        def __init__(self, pipeline_options):
            super().__init__(pipeline_options)
            self.pipeline_options = pipeline_options
            self.enrichment_pipe = [VLMEnricher(enabled=pipeline_options.do_vlm_enrichment)]
            if pipeline_options.do_vlm_enrichment: self.keep_backend = True
        @classmethod
        def get_default_options(cls): return VLMOpts()

    opts = VLMOpts(
        do_ocr=False, generate_picture_images=True, images_scale=3.0,
        do_picture_description=False, do_picture_classification=False,
        do_chart_extraction=False, do_table_structure=True,
        enable_remote_services=True, do_code_enrichment=False,
        do_formula_enrichment=False, do_vlm_enrichment=True
    )
    opts.layout_options = LayoutOptions(confidence_threshold=0.3)
    opts.table_structure_options.mode = TableFormerMode.ACCURATE

    processing_path = tmp_path
    base_name = Path(tmp_path).stem
    ext = suffix.lower()

    if ext in ['.ppt', '.pptx']:
        log("▶ Converting to PDF...")
        subprocess.run(['soffice', '--headless', '--convert-to', 'pdf', tmp_path], check=True)
        processing_path = str(Path(tmp_path).with_suffix(".pdf"))

    log("▶ Running Docling...")
    converter = DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_cls=VLMPipeline, pipeline_options=opts)
    })
    result = converter.convert(processing_path)
    log("✓ Extraction done")

    # Build page-marked markdown
    pages_text = {}
    for element, _ in result.document.iterate_items():
        if not hasattr(element, "prov") or not element.prov: continue
        page_no = getattr(element.prov[0], "page_no", 0)
        if page_no not in pages_text: pages_text[page_no] = []
        if isinstance(element, PictureItem): continue
        if isinstance(element, TableItem):
            md = element.export_to_markdown(doc=result.document)
            if md: pages_text[page_no].append(md)
            continue
        if not (hasattr(element, "text") and element.text): continue
        is_h = isinstance(element, SectionHeaderItem)
        if not is_h and isinstance(element, TextItem):
            if getattr(element, "label", None) in (DocItemLabel.SECTION_HEADER, "section_header"): is_h = True
        pages_text[page_no].append(f"# {element.text.strip()}" if is_h else element.text)

    lines = []
    for pn in sorted(pages_text.keys()):
        lines.append(f"<!-- PAGE {pn} -->")
        lines.extend(pages_text[pn])
    page_marked_md = "\n".join(lines)

    # Extract sections
    page_re   = re.compile(r'^<!-- PAGE (\d+) -->$', re.MULTILINE)
    heading_re = re.compile(r'^#{1,6}\s+(.+)$', re.MULTILINE)
    splits = list(page_re.finditer(page_marked_md))
    sections = {}
    last_title = None
    for i, pm in enumerate(splits):
        pn = int(pm.group(1))
        pt = page_marked_md[pm.end(): splits[i+1].start() if i+1 < len(splits) else len(page_marked_md)]
        hs = list(heading_re.finditer(pt))
        leading = pt[:hs[0].start() if hs else len(pt)].strip()
        if leading and last_title:
            k = (last_title, pn); sections[k] = (sections.get(k,"") + "\n\n" + leading).strip()
        for j, m in enumerate(hs):
            t = m.group(1).strip()
            body = pt[m.end(): hs[j+1].start() if j+1<len(hs) else len(pt)].strip()
            last_title = t; k = (t, pn)
            sections[k] = (sections[k] + "\n\n" + body) if k in sections else body

    # Build tree
    BULLET_CHARS = {"●","•","◦","○","▪","▸","→"}
    def get_prefix(title):
        s = title.strip()
        if not s: return "plain"
        f = s[0]
        if f in BULLET_CHARS: return "bullet"
        if f.isdigit(): return "digit"
        if f.islower(): return "lower_alpha"
        return "plain"

    raw, seen_pages, doc_order = [], set(), 0
    for element, _ in result.document.iterate_items():
        doc_order += 1
        if hasattr(element, "prov") and element.prov:
            for prov in element.prov: seen_pages.add(prov.page_no)
        if isinstance(element, PictureItem): continue
        is_h = isinstance(element, SectionHeaderItem)
        if not is_h and isinstance(element, TextItem):
            if getattr(element, "label", None) in (DocItemLabel.SECTION_HEADER, "section_header"): is_h = True
            elif get_prefix((element.text or "").strip()) in ("bullet","digit","lower_alpha"):
                if len((element.text or "").strip()) <= 60: is_h = True
        if not is_h: continue
        title = (element.text or "").strip()
        if not title: continue
        pn = getattr(element.prov[0], "page_no", 0) if (hasattr(element,"prov") and element.prov) else 0
        raw.append({"title": title, "start": pn, "node_id": str(getattr(element,"self_ref","") or "0000"), "doc_order": doc_order})

    total_pages = max(seen_pages) if seen_pages else 0
    deduped = []
    for entry in raw:
        if deduped and deduped[-1]["title"] == entry["title"]: deduped[-1]["end_page"] = entry["start"]; continue
        if deduped: deduped[-1]["end_page"] = entry["start"] - 1
        deduped.append(entry)
    if deduped: deduped[-1]["end_page"] = total_pages

    # Build heading tree
    roots, stack, dyn, nl = [], [], {}, 1
    for entry in deduped:
        node = dict(entry); ptype = get_prefix(entry["title"])
        if ptype == "plain": level = 0; dyn = {}; nl = 1
        else:
            if ptype not in dyn: dyn[ptype] = nl; nl += 1
            elif stack and dyn[ptype] <= stack[-1][1]:
                if not any(get_prefix(s[0]["title"]) == ptype for s in stack): dyn[ptype] = nl; nl += 1
            level = dyn[ptype]
        while stack and stack[-1][1] >= level: stack.pop()
        if stack: stack[-1][0].setdefault("children", []).append(node)
        else: roots.append(node)
        stack.append((node, level))

    nc = count(1); flat_nodes = []
    def enrich(node):
        title = node["title"]; pn = node.get("start",0); ep = node.get("end_page", pn)
        content = "\n".join(filter(None, [sections.get((title,p),"") for p in range(pn, ep+1)]))
        e = {"title": title, "node_id": f"{next(nc):04d}", "page_index": pn,
             "doc_order": node.get("doc_order",0), "text": f"{title}\n{content}" if content else title}
        ch = [enrich(c) for c in node.get("children",[])]
        if ch: e["children"] = ch
        flat_nodes.append(e); return e

    nodes = [enrich(n) for n in roots]
    def _strip(n):
        n.pop("doc_order", None)
        for c in n.get("children",[]): _strip(c)
    for n in nodes: _strip(n)
    ideal = {"total_pages": total_pages, "nodes": nodes}

    stem = Path(uploaded.name).stem
    tree_path = str(perm_dir / f"{stem}_tree.json")
    with open(tree_path, "w", encoding="utf-8") as f: json.dump(ideal, f, indent=2, ensure_ascii=False)
    log(f"✓ Tree → {tree_path}")

    # Embed
    log("▶ Embedding...")
    def flatten_nodes(nodes_list):
        flat = []
        for n in nodes_list:
            flat.append(n)
            if 'children' in n: flat.extend(flatten_nodes(n['children']))
        return flat

    all_nodes = flatten_nodes(ideal.get("nodes", []))
    for i in range(len(all_nodes)-1): all_nodes[i]["next_index"] = all_nodes[i+1]["page_index"]
    if all_nodes: all_nodes[-1]["next_index"] = None

    dc = 0
    for node in all_nodes:
        chunks = chunk_text(node["text"])
        embs   = get_embeddings(chunks)
        ci     = node["page_index"]
        ni     = node["next_index"] if node["next_index"] is not None else total_pages
        for chunk, emb in zip(chunks, embs):
            dc += 1
            collection.add(ids=[f"doc{dc}"], documents=[chunk], embeddings=[emb],
                           metadatas=[{"node_id": node["node_id"], "curr_index": ci,
                                       "next_index": ni, "filename": f"{stem}_tree.json"}])
    log(f"✓ {dc} chunks indexed")
    st.success(f"✓ Done! {dc} chunks indexed.")


# ── Main layout
col_chat, col_pdf = st.columns([2, 1])

with col_chat:
    st.markdown("### 💬 Ask")
    question = st.text_input("Question", placeholder="What are the key financial planning basics?", label_visibility="collapsed")

    if st.button("Send →") and question:
        trees = sorted(perm_dir.glob("*_tree.json"), key=os.path.getmtime, reverse=True)
        if not trees:
            st.warning("Index a document first.")
            st.stop()

        tree_path = str(trees[0])
        with st.spinner("Thinking..."):
            split_resp = client.chat.completions.create(
                model=VISION_MODEL,
                messages=[{"role": "user", "content":
                    "Split into individual questions. Resolve pronouns. Return JSON array of strings. No question marks. No extra text.\n" + question}]
            )
            raw = split_resp.choices[0].message.content.strip().strip("```json").strip("```").strip()
            queries = json.loads(raw) if raw else [question]

            for q in queries:
                results = collection.query(
                    query_embeddings=get_embeddings([q]),
                    n_results=10,
                    include=["distances", "metadatas", "documents"]
                )
                selected      = get_nodes(results)
                retrieved_docs = [doc for docs in results.get("documents", []) for doc in docs]

                pages_to_fetch = selected[1] if selected[1] else selected[2][:5]
                context = ""
                if pages_to_fetch:
                    context = '\n'.join(filter(None, ext_text(tree_path, pages_to_fetch)))
                if not context.strip():
                    context = "\n\n".join(retrieved_docs[:6])

                sys_prompt = "Strictly refer to given text only. State all relevant data with page numbers. Under 'Inference', explain every possible scenario. Do not make up information."
                answer = client.chat.completions.create(
                    model=VISION_MODEL,
                    messages=[{"role": "user", "content": sys_prompt + "\n\n" + context + "\n\nAnswer: " + q}]
                )
                st.markdown(f"**Q: {q}**")
                st.markdown(answer.choices[0].message.content)
                if selected[2]:
                    st.caption(f"Pages · {', '.join(str(p) for p in selected[2])}")
                st.divider()

with col_pdf:
    if uploaded and Path(uploaded.name).suffix.lower() == ".pdf":
        import fitz
        uploaded.seek(0)
        doc = fitz.open(stream=uploaded.read(), filetype="pdf")
        page_num = st.slider("Page", 1, doc.page_count, 1) - 1
        pix = doc[page_num].get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        st.image(pix.tobytes("png"))
        doc.close()