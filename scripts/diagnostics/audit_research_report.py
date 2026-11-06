#!/usr/bin/env python3
"""Check report anchors, captions and local artifact references without a browser."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote,urlsplit
import argparse
import datetime
import json


class Parser(HTMLParser):
    def __init__(self):
        super().__init__();self.ids=[];self.links=[];self.figures=0;self.captions=0
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        self.figures+=tag=='figure';self.captions+=tag=='figcaption'
        key='href' if tag=='a' else 'src'
        if tag in ('a','img','video','source') and key in a:self.links.append(a[key])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    parser=Parser();parser.feed(a.source.read_text());missing=[];checked=0
    for ref in parser.links:
        u=urlsplit(ref)
        if u.scheme or u.netloc:continue
        if not u.path:
            if u.fragment and u.fragment not in parser.ids:missing.append(ref)
        else:
            checked+=1
            if not (a.source.parent/unquote(u.path)).resolve().exists():missing.append(ref)
    duplicates=sorted({key for key in parser.ids if parser.ids.count(key)>1})
    result={'passed':not missing and not duplicates and parser.figures==parser.captions,
            'audited':datetime.datetime.now().isoformat(timespec='seconds'),
            'figures':parser.figures,'captions':parser.captions,'local_references_checked':checked,
            'missing':missing,'duplicate_ids':duplicates,
            'scope':'HTML structural and artifact-link validation only. Fresh browser layout validation remains blocked by the renderer SVGPoint tool error.'}
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result));assert result['passed']


if __name__=='__main__':main()
