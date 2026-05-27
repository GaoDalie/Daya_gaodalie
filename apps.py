import os
import json
import io
import base64
import re
import tempfile
import subprocess
import chromadb
from pathlib import Path
from collections import defaultdict
from itertools import count

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="DAYA API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
client         = OpenAI(api_key=OPENAI_API_KEY)
VISION_MODEL   = "gpt-4o"
EMBED_MODEL    = "text-embedding-3-small"
CHUNK_SIZE     = 512
CHUNK_OVERLAP  = 16
PERM_DIR       = Path.home() / ".daya"
PERM_DIR.mkdir(parents=True, exist_ok=True)

# ── DB
def get_collection():
    chroma_path = PERM_DIR / "chroma"
    chroma_path.mkdir(parents=True, exist_ok=True)
    db = chromadb.PersistentClient(path=str(chroma_path))
    return db.get_or_create_collection(name="daya_demo")

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
    filtered  = [(d, m) for d, m in zip(dist, metadatas) if d <= 1.2]
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
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    def flatten(nodes):
        flat = []
        for n in nodes:
            flat.append(n)
            if "children" in n: flat.extend(flatten(n["children"]))
        return flat
    all_nodes = flatten(data.get("nodes", []))
    result = []
    for page_num in page_list:
        content = [n.get("text", "") for n in all_nodes if n.get("page_index") == page_num]
        result.append("\n".join(filter(None, content)).strip())
    return result


# ══════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"name": "DAYA API", "version": "1.0", "status": "running"}


@app.get("/status")
def status():
    trees = sorted(PERM_DIR.glob("*_tree.json"), key=os.path.getmtime, reverse=True)
    col   = get_collection()
    count = col.count()
    return {
        "indexed": count > 0,
        "chunks":  count,
        "document": trees[0].name.replace("_tree.json", "") if trees else None,
    }


@app.post("/index")
async def index_document(file: UploadFile = File(...)):
    """Upload and index a PDF, PPTX, or DOCX file."""

    suffix = Path(file.filename).suffix.lower()
    if suffix not in [".pdf", ".pptx", ".ppt", ".docx"]:
        raise HTTPException(400, "Unsupported file type. Use PDF, PPTX, or DOCX.")

    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    from docling_core.types.doc.document import PictureItem, PictureDescriptionData, TableItem
    from docling.datamodel.base_models import InputFormat, ItemAndImageEnrichmentElement
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.models.base_model import BaseItemAndImageEnrichmentModel
    from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, LayoutOptions
    from docling_core.types.doc import SectionHeaderItem, TextItem
    from docling_core.types.doc.labels import DocItemLabel

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
    if suffix in [".ppt", ".pptx"]:
        subprocess.run(["soffice", "--headless", "--convert-to", "pdf", tmp_path], check=True)
        processing_path = str(Path(tmp_path).with_suffix(".pdf"))

    converter = DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_cls=VLMPipeline, pipeline_options=opts)
    })
    result = converter.convert(processing_path)

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
    page_re    = re.compile(r"^<!-- PAGE (\d+) -->$", re.MULTILINE)
    heading_re = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)
    splits     = list(page_re.finditer(page_marked_md))
    sections   = {}
    last_title = None
    for i, pm in enumerate(splits):
        pn = int(pm.group(1))
        pt = page_marked_md[pm.end(): splits[i+1].start() if i+1 < len(splits) else len(page_marked_md)]
        hs = list(heading_re.finditer(pt))
        leading = pt[:hs[0].start() if hs else len(pt)].strip()
        if leading and last_title:
            k = (last_title, pn); sections[k] = (sections.get(k, "") + "\n\n" + leading).strip()
        for j, m in enumerate(hs):
            t    = m.group(1).strip()
            body = pt[m.end(): hs[j+1].start() if j+1 < len(hs) else len(pt)].strip()
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

    stem      = Path(file.filename).stem
    tree_path = str(PERM_DIR / f"{stem}_tree.json")
    with open(tree_path, "w", encoding="utf-8") as f:
        json.dump(ideal, f, indent=2, ensure_ascii=False)

    # Embed
    def flatten_nodes(nodes_list):
        flat = []
        for n in nodes_list:
            flat.append(n)
            if "children" in n: flat.extend(flatten_nodes(n["children"]))
        return flat

    all_nodes = flatten_nodes(ideal.get("nodes", []))
    for i in range(len(all_nodes)-1): all_nodes[i]["next_index"] = all_nodes[i+1]["page_index"]
    if all_nodes: all_nodes[-1]["next_index"] = None

    col = get_collection()
    dc  = 0
    for node in all_nodes:
        chunks = chunk_text(node["text"])
        embs   = get_embeddings(chunks)
        ci     = node["page_index"]
        ni     = node["next_index"] if node["next_index"] is not None else total_pages
        for chunk, emb in zip(chunks, embs):
            dc += 1
            col.add(ids=[f"doc{dc}"], documents=[chunk], embeddings=[emb],
                    metadatas=[{"node_id": node["node_id"], "curr_index": ci,
                                "next_index": ni, "filename": f"{stem}_tree.json"}])

    return {"status": "indexed", "document": stem, "chunks": dc, "pages": total_pages}


class URLRequest(BaseModel):
    url: str


@app.post("/index-url")
async def index_from_url(req: URLRequest):
    """Fetch a PDF from a URL and index it."""
    import httpx

    async with httpx.AsyncClient(timeout=60) as http:
        response = await http.get(req.url)
        if response.status_code != 200:
            raise HTTPException(400, f"Could not fetch URL: {response.status_code}")

    content_type = response.headers.get("content-type", "")
    if "pdf" in content_type:
        suffix = ".pdf"
    else:
        # Guess from URL
        suffix = Path(req.url.split("?")[0]).suffix.lower() or ".pdf"

    filename = Path(req.url.split("?")[0]).name or "document.pdf"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name

    # Reuse the same indexing logic by creating a fake UploadFile-like object
    class FakeUpload:
        def __init__(self, path, name):
            self.filename = name
            self._path = path
        async def read(self):
            return open(self._path, "rb").read()

    fake = FakeUpload(tmp_path, filename)
    return await index_document(fake)





@app.post("/query")
def query(req: QueryRequest):
    """Ask a question about the indexed document."""

    trees = sorted(PERM_DIR.glob("*_tree.json"), key=os.path.getmtime, reverse=True)
    if not trees:
        raise HTTPException(404, "No document indexed. POST to /index first.")

    tree_path = str(trees[0])

    split_resp = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{"role": "user", "content":
            "Split into individual questions. Resolve pronouns. Return JSON array of strings. No question marks. No extra text.\n" + req.question}]
    )
    raw_split = split_resp.choices[0].message.content.strip().strip("```json").strip("```").strip()
    queries   = json.loads(raw_split) if raw_split else [req.question]

    col     = get_collection()
    answers = []

    for q in queries:
        results = col.query(
            query_embeddings=get_embeddings([q]),
            n_results=10,
            include=["distances", "metadatas", "documents"]
        )
        selected       = get_nodes(results)
        retrieved_docs = [doc for docs in results.get("documents", []) for doc in docs]

        pages_to_fetch = selected[1] if selected[1] else selected[2][:5]
        context = ""
        if pages_to_fetch:
            context = "\n".join(filter(None, ext_text(tree_path, pages_to_fetch)))
        if not context.strip():
            context = "\n\n".join(retrieved_docs[:6])

        sys_prompt = "Strictly refer to given text only. State all relevant data with page numbers. Under 'Inference', explain every possible scenario. Do not make up information. Reply 'I don't know the answer' if information is absent."
        answer = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{"role": "user", "content": sys_prompt + "\n\n" + context + "\n\nAnswer: " + q}]
        )
        answers.append({
            "question": q,
            "answer":   answer.choices[0].message.content,
            "pages":    selected[2],
        })

    return {"results": answers}