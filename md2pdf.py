import sys, os, markdown
from xhtml2pdf import pisa
md_path, pdf_path, base = sys.argv[1], sys.argv[2], sys.argv[3]
text=open(md_path,encoding='utf-8').read()
body=markdown.markdown(text, extensions=['tables','fenced_code','sane_lists'])
html=f"""<html><head><meta charset="utf-8"><style>
@page {{ size: A4; margin: 1.8cm; }}
body{{font-family:Helvetica,Arial,sans-serif;font-size:10.5px;line-height:1.45;color:#111;}}
h1{{font-size:19px;border-bottom:1.5px solid #888;padding-bottom:4px;}}
h2{{font-size:14px;margin-top:15px;color:#000;}}
h3{{font-size:12px;color:#222;margin-top:10px;}}
table{{border-collapse:collapse;width:100%;font-size:9.5px;margin:6px 0;}}
th,td{{border:1px solid #999;padding:4px 6px;text-align:left;}} th{{background:#eee;}}
img{{max-width:100%;margin:6px 0;}} code{{background:#f2f2f2;padding:1px 2px;}}
em{{color:#444;}}
</style></head><body>{body}</body></html>"""
def link_callback(uri, rel):
    if uri.startswith('http'): return uri
    p=os.path.join(base, uri)
    return p if os.path.exists(p) else uri
with open(pdf_path,'wb') as f:
    err=pisa.CreatePDF(html, dest=f, link_callback=link_callback)
print("PDF errors:", err.err if hasattr(err,'err') else err)
print("wrote", pdf_path, os.path.getsize(pdf_path), "bytes")
