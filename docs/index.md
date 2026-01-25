# Documentation

This file exists primarily so Read the Docs generates an `index.html` at the HTML output root.

```{only} html and docs_en
```{raw} html
<meta http-equiv="refresh" content="0; url=en/index.html" />
<link rel="canonical" href="en/index.html" />
<p>Redirecting to the English documentation… If you are not redirected, open <a href="en/index.html">en/index.html</a>.</p>
```
```

```{only} html and docs_ja
```{raw} html
<meta http-equiv="refresh" content="0; url=ja/index.html" />
<link rel="canonical" href="ja/index.html" />
<p>日本語ドキュメントへ移動します… 自動的に移動しない場合は <a href="ja/index.html">ja/index.html</a> を開いてください。</p>
```
```

```{only} not html and docs_en
See the English documentation at: `en/index`.
```

```{only} not html and docs_ja
日本語ドキュメント: `ja/index`.
```

```{ifconfig} language == 'en'
```{toctree}
:hidden:

en/index
```
```

```{ifconfig} language == 'ja'
```{toctree}
:hidden:

ja/index
```
```
