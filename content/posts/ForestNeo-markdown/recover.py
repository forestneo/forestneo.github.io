#!/usr/bin/env python3
"""Recover Hexo article HTML, including MathJax and highlight.js, as Markdown.

Requires lxml. Run with the bundled Codex Python documented in README.md.
The original HTML is never modified. Recovery cannot reproduce source whitespace.
"""

import argparse
import base64
from collections import Counter
import hashlib
from html import escape
import json
import os
from pathlib import Path
import re
import shutil
from urllib.parse import unquote, urljoin, urlsplit

from lxml import html


def has_class(node, name):
    return name in node.get("class", "").split()


def select_class(root, name):
    return root.xpath(
        './/*[contains(concat(" ", normalize-space(@class), " "), $name)]',
        name=" " + name + " ",
    )


def plain(node):
    return " ".join(node.text_content().split()) if node is not None else ""


def destination(url):
    return "<" + url.replace("<", "%3C").replace(">", "%3E").replace("\n", "%0A") + ">"


def json_value(value):
    # JSON strings and arrays are also valid YAML values.
    return json.dumps(value, ensure_ascii=False)


def normalized_article_path(path):
    path = path.removesuffix("index.html").rstrip("/")
    parent, _, slug = path.rpartition("/")
    slug = re.sub(r"^(?:PAPER|论文阅读)[- ]*", "", slug, flags=re.I)
    return parent + "/" + re.sub(r"[\W_]", "", slug).casefold()


def strip_promotional_footer(markdown):
    """Remove the old account footer, leaving body rules and fenced code intact."""
    rules = []
    fence = None
    offset = 0
    for line in markdown.splitlines(keepends=True):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
        elif fence is None and re.fullmatch(r"[ \t]*---[ \t]*", line.rstrip("\r\n")):
            rules.append((offset, offset + len(line)))
        offset += len(line)
    if rules:
        start, end = rules[-1]
        tail = markdown[end:].lstrip()
        if tail.startswith("本篇内容到这里就结束了") and "公众号" in tail.split("\n", 1)[0]:
            return markdown[:start].rstrip(), "separator_and_promotion"

    summary = re.search(
        r'(?:<a id="总结"></a>\s*)?^# 总结\s+本章[^\n]*下面是我的公众号二维码[^\n]*\s+'
        r'!\[[^\]]*\]\(<https://forest-pic\.oss-cn-beijing\.aliyuncs\.com/20200308122411\.png>\)\s*\Z',
        markdown, re.M,
    )
    if summary:
        return markdown[:summary.start()].rstrip(), "promotion_only_summary"
    qr = re.search(
        r'^!\[[^\]]*\]\(<https://forest-pic\.oss-cn-beijing\.aliyuncs\.com/20200308122411\.png>\)\s*\Z',
        markdown, re.M,
    )
    if qr:
        return markdown[:qr.start()].rstrip(), "trailing_qr_code"
    return markdown, None


class Converter:
    def __init__(self, source, output, source_root, output_root, post_map, original_url, deleted_targets=None):
        self.source = source
        self.output = output
        self.source_root = source_root
        self.output_root = output_root
        self.post_map = post_map
        self.original_url = original_url
        self.deleted_targets = deleted_targets or set()
        self.tokens = {}
        self.coverage = set()
        self.stats = Counter()
        self.formulas = []
        self.code_blocks = []
        self.image_urls = []
        self.warnings = []
        self.rewritten_links = []
        self.inline_math = False

    def protect(self, value):
        token = f"\ue000RECOVER{len(self.tokens)}\ue001"
        self.tokens[token] = value
        return token

    def expand(self, value):
        # Some tokens (e.g. a blockquote containing code) refer to earlier ones.
        for token, content in reversed(list(self.tokens.items())):
            value = value.replace(token, content)
        return value

    def text(self, node, field):
        self.coverage.add((node, field))
        value = getattr(node, field) or ""
        value = re.sub(r"\s+", " ", value)
        # Hexo can split $...$ across <em> nodes at superscript stars.
        # Carry math state across adjacent text nodes before escaping prose.
        pieces = []
        for index, character in enumerate(value):
            if character == "$" and (index == 0 or value[index - 1] != "\\"):
                self.inline_math = not self.inline_math
                pieces.append(character)
            elif not self.inline_math and character in "\\`*_[]<>#":
                pieces.append("\\" + character)
            else:
                pieces.append(character)
        result = "".join(pieces)
        # Literal numbered lines must not become new Markdown lists.
        result = re.sub(r"^(\s*\d+)([.)])(?=\s|$)", r"\1\\\2", result)
        return re.sub(r"^(\s*)([-+])(?=\s)", r"\1\\\2", result)

    def children(self, node):
        pieces = [self.text(node, "text")]
        for child in node:
            pieces.append(self.render(child))
            pieces.append(self.text(child, "tail"))
        return "".join(pieces)

    def raw_text(self, node):
        self.coverage.add((node, "text"))
        pieces = [node.text or ""]
        for child in node:
            pieces.append("\n" if child.tag == "br" else self.raw_text(child))
            self.coverage.add((child, "tail"))
            pieces.append(child.tail or "")
        return "".join(pieces)

    def mark_subtree(self, node):
        for descendant in node.iter():
            self.coverage.add((descendant, "text"))
            if descendant is not node:
                self.coverage.add((descendant, "tail"))

    def url(self, url, image=False):
        if url.startswith("data:"):
            if not image:
                return url
            try:
                header, encoded = url.split(",", 1)
                data = base64.b64decode(encoded) if ";base64" in header else unquote(encoded).encode()
                extension = ".png" if data.startswith(b"\x89PNG") else ".gif"
                target = self.output_root / "assets" / (hashlib.sha256(data).hexdigest()[:16] + extension)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                self.stats["embedded_images"] += 1
                self.warnings.append({"kind": "embedded_image", "detail": "原网页内嵌的 1×1 占位图片已原样保存。"})
                return Path(os.path.relpath(target, self.output.parent)).as_posix()
            except (ValueError, TypeError):
                return url
        if url.startswith("#"):
            return url
        absolute = urljoin(self.original_url, url)
        parsed = urlsplit(absolute)
        if parsed.scheme not in ("http", "https"):
            return url
        if parsed.hostname not in {urlsplit(self.original_url).hostname, "forestneo.top", "www.forestneo.top", "forestneo.topcom", "forestneo.com", "www.forestneo.com", "forestneo.github.io"}:
            return "https:" + url if url.startswith("//") else url
        path = unquote(parsed.path)
        key = path.removesuffix("index.html").rstrip("/") + "/"
        target = self.post_map.get(key)
        if not image and target is None:
            # Older article links use different separators / PAPER prefixes.
            aliases = {target for original, target in self.post_map.items() if normalized_article_path(original) == normalized_article_path(key)}
            if len(aliases) == 1:
                target = aliases.pop()
            elif not aliases:
                # Some old links use only the paper acronym (e.g. PrivKV).
                normalized = normalized_article_path(key)
                if len(normalized.rsplit("/", 1)[-1]) >= 5:
                    prefixes = {target for original, target in self.post_map.items()
                                if normalized_article_path(original).startswith(normalized)}
                    if len(prefixes) == 1:
                        target = prefixes.pop()
        if not image and target is not None:
            if target in self.deleted_targets:
                return None  # Keep the reference text without a broken link.
            relative = Path(os.path.relpath(target, self.output.parent)).as_posix()
            self.stats["rewritten_article_links"] += 1
            # Hexo excerpt anchors are not part of the saved article body.
            fragment = parsed.fragment if parsed.fragment != "more" else ""
            result = relative + ("?" + parsed.query if parsed.query else "") + ("#" + fragment if fragment else "")
            self.rewritten_links.append({"original": url, "recovered": result})
            return result
        if image and url.startswith("/Users/"):
            candidate = Path(unquote(url))
        else:
            candidate = (self.source_root / path.lstrip("/")).resolve()
        if candidate.is_file() and (candidate.is_relative_to(self.source_root) or url.startswith("/Users/")):
            asset_path = candidate.relative_to(self.source_root) if candidate.is_relative_to(self.source_root) else Path("local") / candidate.name
            target = self.output_root / "assets" / asset_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, target)
            return Path(os.path.relpath(target, self.output.parent)).as_posix()
        if image:
            self.warnings.append({"kind": "missing_local_image", "url": url, "detail": "源目录没有对应图片，保留原地址。"})
            return url if url.startswith("/Users/") else absolute
        return absolute

    def code(self, node):
        code_cells = node.xpath('.//td[contains(concat(" ", normalize-space(@class), " "), " code ")]')
        code_node = code_cells[0] if code_cells else node
        lines = code_node.xpath('.//span[contains(concat(" ", normalize-space(@class), " "), " line ")]')
        content = "\n".join(self.raw_text(line) for line in lines) if lines else self.raw_text(code_node).rstrip("\n")
        self.mark_subtree(node)
        self.code_blocks.append(content)
        classes = node.get("class", "").split()
        language = next((c for c in classes if c not in {"highlight", "plain"}), "")
        if node.tag == "pre":
            code = node.find("code")
            if code is not None:
                language = next((c.removeprefix("language-") for c in code.get("class", "").split() if c.startswith("language-")), "")
        longest = max((len(run) for run in re.findall(r"`+", content)), default=0)
        fence = "`" * max(3, longest + 1)
        self.stats["code_blocks"] += 1
        return "\n\n" + self.protect(f"{fence}{language}\n{content}\n{fence}") + "\n\n"

    def table(self, node):
        rows = node.xpath("./tr|./thead/tr|./tbody/tr|./tfoot/tr")
        matrix = []
        alignments = []
        for row in rows:
            values = []
            for cell in row.xpath("./th|./td"):
                self.inline_math = False
                value = self.expand(self.children(cell)).strip()
                self.inline_math = False
                value = re.sub(r"\s*\n\s*", "<br>", value).replace("\ue002", "<br>").replace("|", "\\|")
                values.append(value)
                if not matrix:
                    alignment = cell.get("align", "")
                    alignments.append({"center": ":---:", "right": "---:", "left": ":---"}.get(alignment, "---"))
            matrix.append(values)
        width = max(map(len, matrix), default=0)
        if not width:
            return ""
        matrix = [row + [""] * (width - len(row)) for row in matrix]
        # A headerless table must not silently promote its first data row.
        has_header = bool(rows[0].xpath("./th"))
        header = matrix.pop(0) if has_header else [""] * width
        separator = (alignments + ["---"] * width)[:width]
        result = ["| " + " | ".join(row) + " |" for row in [header, separator] + matrix]
        self.stats["tables"] += 1
        return "\n\n" + self.protect("\n".join(result)) + "\n\n"

    def render(self, node):
        if not isinstance(node.tag, str):
            return "\n\n<!-- more -->\n\n" if "more" in (node.text or "").strip().lower() else ""
        tag = node.tag.lower()
        if has_class(node, "headerlink") or has_class(node, "gutter") or tag == "style":
            return ""
        if tag == "script":
            if node.get("type", "").startswith("math/tex"):
                latex = self.raw_text(node).strip()
                self.formulas.append(latex)
                self.stats["formulas"] += 1
                display = "mode=display" in node.get("type", "")
                return "\n\n" + self.protect("$$\n" + latex + "\n$$") + "\n\n" if display else self.protect("$" + latex + "$")
            raise ValueError("Unexpected non-math script in article body")
        if (tag == "figure" and has_class(node, "highlight")) or tag == "pre":
            return self.code(node)
        if tag == "img":
            original = node.get("data-original") or node.get("data-src") or node.get("src", "")
            url = self.url(original, image=True)
            self.image_urls.append(url)
            self.stats["images"] += 1
            alt = re.sub(r"([\\\[\]])", r"\\\1", node.get("alt", ""))
            return self.protect(f"![{alt}]({destination(url)})")
        if tag == "br":
            return "\ue002"  # Hard break, kept through whitespace normalization.
        if tag == "hr":
            return "\n\n---\n\n"
        if tag == "table":
            return self.table(node)
        if tag == "code":
            content = self.raw_text(node)
            fence = "`" * (1 + max((len(run) for run in re.findall(r"`+", content)), default=0))
            padding = " " if content.startswith(("`", " ")) or content.endswith(("`", " ")) else ""
            return self.protect(f"{fence}{padding}{content}{padding}{fence}")
        if tag == "sup":
            self.mark_subtree(node)
            return self.protect(html.tostring(node, encoding="unicode", with_tail=False))
        if tag == "configuration":
            self.mark_subtree(node)
            self.warnings.append({"kind": "literal_html", "detail": "原文未转义的 configuration 标签恢复为行内代码。"})
            return self.protect("`" + html.tostring(node, encoding="unicode", with_tail=False) + "`")
        if tag in {"ul", "ol"}:
            self.stats["lists"] += 1
            self.text(node, "text")
            items = []
            for index, item in enumerate(node.xpath("./li"), int(node.get("start", "1"))):
                self.inline_math = False
                value = self.normalize(self.children(item))
                self.inline_math = False
                self.text(item, "tail")
                marker = f"{index}. " if tag == "ol" else "- "
                lines = value.splitlines() or [""]
                items.append(marker + lines[0] + "".join("\n" + " " * len(marker) + line if line else "\n" for line in lines[1:]))
            return "\n\n" + self.protect("\n\n".join(items)) + "\n\n"
        block = tag in {"p", "div", "center", "section", "figure", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6"}
        if block:
            self.inline_math = False
        content = self.children(node)
        if block:
            self.inline_math = False
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            anchor = f'<a id="{escape(node.get("id"), quote=True)}"></a>\n\n' if node.get("id") else ""
            self.stats["headings"] += 1
            return "\n\n" + self.protect(anchor) + "#" * int(tag[1]) + " " + content.strip() + "\n\n"
        if tag in {"strong", "b"}:
            return "**" + content.strip() + "**"
        if tag in {"em", "i"}:
            return "*" + content.strip() + "*"
        if tag in {"del", "s", "strike"}:
            return "~~" + content.strip() + "~~"
        if tag == "a":
            anchor = self.protect(f'<a id="{escape(node.get("id"), quote=True)}"></a>') if node.get("id") else ""
            if not node.get("href"):
                return anchor + content
            self.stats["links"] += 1
            url = self.url(node.get("href"))
            if url is None:
                return anchor + content
            return anchor + "[" + content.strip() + "](" + self.protect(destination(url)) + ")"
        if tag in {"p", "div", "center", "section", "figure"}:
            return "\n\n" + content.strip() + "\n\n" if content.strip() else ""
        if tag == "blockquote":
            value = self.normalize(content)
            anchor = f'<a id="{escape(node.get("id"), quote=True)}"></a>\n\n' if node.get("id") else ""
            return "\n\n" + self.protect(anchor + "\n".join("> " + line if line else ">" for line in value.splitlines())) + "\n\n"
        if tag == "span" and node.get("id") == "more":
            return "\n\n<!-- more -->\n\n" + content
        return content

    def normalize(self, value):
        value = re.sub(r" *\n *", "\n", value)
        value = re.sub(r"\n{3,}", "\n\n", value).strip()
        return self.expand(value).replace("\ue002", "  \n")

    def validate(self, body, markdown):
        missing = []
        for node in body.iter():
            if not isinstance(node.tag, str):
                continue
            skipped = any(has_class(n, "gutter") or has_class(n, "headerlink") or n.tag == "style" for n in [node, *node.iterancestors()])
            if not skipped and (node.text or "").strip() and (node, "text") not in self.coverage:
                missing.append((node.tag, "text", node.text[:100]))
            if node is not body and (node.tail or "").strip() and (node, "tail") not in self.coverage:
                # Whitespace around code highlighter internals is presentation only.
                if not any(has_class(n, "highlight") for n in node.iterancestors()):
                    missing.append((node.tag, "tail", node.tail[:100]))
        assert not missing, f"Unconverted body text: {missing}"
        # Remove only Markdown container prefixes when checking fenced literals.
        literal_blocks = []
        lines = markdown.splitlines()
        index = 0
        while index < len(lines):
            match = re.fullmatch(r"([ >]*)(`{3,}|\$\$)([^`]*)", lines[index])
            if not match:
                index += 1
                continue
            prefix, fence, _ = match.groups()
            block = []
            index += 1
            while index < len(lines) and lines[index].strip().removeprefix(">").strip() != fence:
                line = lines[index]
                block.append(line[len(prefix):] if line.startswith(prefix) else "" if line.strip() == prefix.strip() else line)
                index += 1
            literal_blocks.append("\n".join(block))
            index += 1
        assert all(formula in markdown or formula in literal_blocks for formula in self.formulas), "Formula source lost"
        assert all(code in literal_blocks for code in self.code_blocks), "Code source lost"
        assert all(destination(url) in markdown for url in self.image_urls), "Image lost"
        expected = {
            "formulas": len(body.xpath('.//script[starts-with(@type,"math/tex")]')),
            "images": len(body.xpath(".//img")),
            "code_blocks": len(body.xpath('.//figure[contains(@class,"highlight")]|.//pre[not(ancestor::figure[contains(@class,"highlight")])]')),
            "tables": len(body.xpath('.//table[not(ancestor::figure[contains(@class,"highlight")])]')),
            "headings": len(body.xpath(".//h1|.//h2|.//h3|.//h4|.//h5|.//h6")),
            "lists": len(body.xpath(".//ul|.//ol")),
        }
        assert all(self.stats[key] == count for key, count in expected.items()), (dict(self.stats), expected)
        assert not re.search("[\ue000-\ue002]", markdown), "Unexpanded conversion token"
        return {"text_coverage": "passed", "latex_verbatim": "passed", "code_verbatim": "passed", "element_counts": "passed", "expected": expected}


def main():
    script = Path(__file__).resolve()
    project = next((parent for parent in script.parents
                    if (parent / "ForestNeo-website-master").is_dir()), script.parent.parent)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=project / "ForestNeo-website-master")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate recovered files")
    args = parser.parse_args()
    source_root = args.source.resolve()
    output_root = args.output.resolve()
    assert source_root != output_root and not output_root.is_relative_to(source_root)
    sources = sorted(source_root.glob("20*/*/*/*/index.html"))
    assert sources, "No dated article pages found"
    deletion_manifest = script.parent / "deleted-articles.json"
    deleted = json.loads(deletion_manifest.read_text(encoding="utf-8"))["deleted"] if deletion_manifest.exists() else []
    deleted_files = {entry["file"] for entry in deleted}
    deleted_targets = set()
    outputs = {}
    post_map = {}
    for source in sources:
        parts = source.relative_to(source_root).parts
        date = "-".join(parts[:3])
        target = output_root / "articles" / parts[0] / f"{date}-{parts[3]}.md"
        post_map["/" + "/".join(parts[:-1]) + "/"] = target
        if target.relative_to(output_root).as_posix() in deleted_files:
            deleted_targets.add(target)
            continue
        assert args.overwrite or not target.exists(), f"Output already exists: {target}"
        outputs[source] = target
    assert len(set(post_map.values())) == len(sources), "Output filename collision"
    entries = []
    totals = Counter()
    for source, target in outputs.items():
        document = html.fromstring(source.read_bytes())
        bodies = select_class(document, "post-body")
        assert len(bodies) == 1, f"Expected exactly one article body: {source}"
        body = bodies[0]
        header = select_class(document, "post-header")[0]
        title = plain(select_class(header, "post-title")[0])
        dates = header.xpath('.//time[contains(@itemprop,"dateCreated")]/@datetime')
        updated = header.xpath('.//time[@itemprop="dateModified"]/@datetime')
        categories = [plain(n) for n in select_class(header, "post-category") for n in n.xpath('.//a')]
        tag_sections = select_class(document, "post-tags")
        tags = [plain(n) for section in tag_sections for n in section.xpath('.//a[@rel="tag"]')]
        originals = document.xpath('//link[@itemprop="mainEntityOfPage"]/@href')
        original = originals[0] if originals else "http://forestneo.top/" + source.parent.relative_to(source_root).as_posix() + "/"
        authors = document.xpath('//span[@itemprop="author"]/meta[@itemprop="name"]/@content')
        metadata = {"title": title, "date": dates[0], "updated": updated[0] if updated else dates[0], "categories": categories, "tags": tags}
        if authors:
            metadata["author"] = authors[0]
        metadata["original_url"] = original
        metadata["source_html"] = source.relative_to(project).as_posix() if source.is_relative_to(project) else str(source)
        converter = Converter(source, target, source_root, output_root, post_map, original, deleted_targets)
        markdown = converter.normalize(converter.children(body))
        assert markdown.strip(), f"Empty body: {source}"
        verification = converter.validate(body, markdown)
        # Validate complete source recovery first, then apply requested editorial cleanup.
        markdown, footer_removed = strip_promotional_footer(markdown)
        if converter.stats["formulas"] or re.search(r"\$[^$]+\$", markdown):
            metadata["mathjax"] = True
        frontmatter = "---\n" + "\n".join(f"{key}: {json_value(value)}" for key, value in metadata.items()) + "\n---\n\n"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(frontmatter + markdown + "\n", encoding="utf-8")
        totals.update(converter.stats)
        entries.append({"file": target.relative_to(output_root).as_posix(), "metadata": metadata, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "stats": dict(converter.stats), "validation": verification, "footer_removed": footer_removed, "rewritten_links": converter.rewritten_links, "warnings": converter.warnings})
    report = {"article_count": len(entries), "counts": dict(totals), "validation": "passed", "articles": entries}
    (output_root / "recovery-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dp_entries = [entry for entry in entries if set(entry["metadata"]["tags"]) & {"DP", "LDP", "Differential Privacy", "Local Differential Privacy"}]
    warning_entries = [entry for entry in entries if entry["warnings"]]
    readme = ["# ForestNeo 旧博客 Markdown 恢复", "", f"从 Hexo 发布目录 `ForestNeo-website-master` 恢复了 **{len(entries)} 篇文章**（2017—2023 年），其中 **{len(dp_entries)} 篇带 DP / LDP 相关标签**。", "", "文章按年份保存在 `articles/`，文件名使用原日期和原网址目录名，避免同名文章互相覆盖。每篇保留标题、发布日期、更新日期、分类、标签、作者和原网址。", "", "正文恢复了标题层级、列表、引用、表格、链接、图片、代码和 LaTeX 公式。代码高亮的行号已去除；站内文章链接已指向本地恢复文件；原有标题锚点保留为少量 HTML，供目录和交叉链接跳转。", "", "这是根据发布网页恢复的内容，原 Markdown 的空白、注释、排版选择及未发布内容无法从网页还原。正文中已有的笔误、重复标题、TODO 和未完成段落保留原样。远程图片保留原网址，未检查远程地址是否仍可访问。", "", "## 校验", "", f"所有 {len(entries)} 篇均通过正文文本覆盖校验；{totals['formulas']} 处 MathJax 公式及 {totals['code_blocks']} 个代码块通过原文逐字校验，{totals['tables']} 张表格、{totals['images']} 个图片引用及标题、列表数量核对通过。明细见 `recovery-report.json`。", "", "## 源文件中的图片问题", "", "下列问题来自原网页，具体地址保存在校验报告中：", ""]
    footer_count = sum(bool(entry["footer_removed"]) for entry in entries)
    readme[readme.index("## 源文件中的图片问题"):readme.index("## 源文件中的图片问题")] = ["## 文末清理", "", f"完整恢复校验后，已删除 {footer_count} 篇文章末尾的公众号宣传、二维码及其分隔线。报告中的原文统计和校验对应清理前内容，`footer_removed` 记录清理类型。重新生成时也会应用此清理。", ""]
    for entry in warning_entries:
        counts = Counter(w["kind"] for w in entry["warnings"])
        descriptions = []
        if counts["missing_local_image"]:
            descriptions.append(f"{counts['missing_local_image']} 张本地图片在源目录中缺失，原地址保留")
        if counts["embedded_image"]:
            descriptions.append(f"{counts['embedded_image']} 张图片原本就是 1×1 内嵌占位图，已保存到 assets")
        if counts["literal_html"]:
            descriptions.append("未转义的 configuration 标签转为行内代码")
        readme.append(f"- [{entry['metadata']['title']}]({destination(entry['file'])})：{'；'.join(descriptions)}。")
    readme += ["", "## DP / LDP 笔记", ""]
    for entry in dp_entries:
        readme.append(f"- [{entry['metadata']['title']}]({destination(entry['file'])})")
    readme += ["", "## 全部文章", ""]
    for year in sorted({entry["file"].split("/")[1] for entry in entries}):
        subset = [entry for entry in entries if entry["file"].split("/")[1] == year]
        readme += [f"### {year}（{len(subset)} 篇）", ""]
        for entry in subset:
            readme.append(f"- {entry['metadata']['date'][:10]} · [{entry['metadata']['title']}]({destination(entry['file'])})")
        readme.append("")
    runtime = "/Users/sunlin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
    script_path = script.relative_to(project).as_posix() if script.is_relative_to(project) else str(script)
    readme += ["## 重新生成", "", "`recover.py` 需要 Python 和 lxml。可使用本机 Codex 自带的 Python，从项目根目录执行：", "", "```sh", f"{runtime} {script_path} --overwrite", "```", "", "该命令只重新生成恢复目录中的文章和报告，原 HTML 不会被修改。", ""]
    (output_root / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(json.dumps({"output": str(output_root), "article_count": len(entries), "dp_ldp_count": len(dp_entries), "counts": dict(totals), "validation": "passed", "warning_articles": len(warning_entries)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
