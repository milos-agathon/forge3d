# Unicode bidi conformance data

Version: Unicode 17.0.0.

Official sources:

- `https://www.unicode.org/Public/17.0.0/ucd/BidiCharacterTest.txt`
  SHA-256 `a3e6e905ab5afbe318a96df5401d0372a04cd73ef139ab5e3cf0ae241c255488`
- `https://www.unicode.org/Public/17.0.0/ucd/BidiTest.txt`
  SHA-256 `888bdfc8090652272d1f859cdb00ae659e2dc6c26740be61ef1d03998a687620`
- `https://www.unicode.org/Public/17.0.0/ucd/BidiBrackets.txt`
  SHA-256 `dadbaf38a0d0246e5b805bf8725cb81b7c621f93d030595635f5ba2c2f179428`
  (generated into `src/labels/shape/bidi_brackets.rs`)
- `https://www.unicode.org/Public/17.0.0/ucd/auxiliary/LineBreakTest.txt`
  SHA-256 `e69884e0dde6a8724873f885d68c52dc14518abf9ae4ca9e2283b8773db3b752`
- `https://www.unicode.org/Public/17.0.0/ucd/emoji/emoji-data.txt`
  SHA-256 `2cb2bb9455cda83e8481541ecf5b6dfda66a3bb89efa3fa7c5297eccf607b72b`
  (reserved Extended_Pictographic ranges generated into
  `src/labels/shape/linebreak_emoji.rs` for rule LB30b)

The Unicode License v3 notice distributed at
`src/labels/unicode/LICENSE-UNICODE` applies to these files.

Regenerate both derived Rust tables with:

```text
python src/labels/shape/generate_unicode_tables.py --bidi-brackets <ucd>/BidiBrackets.txt --emoji-data <ucd>/emoji/emoji-data.txt --output-dir src/labels/shape
```
