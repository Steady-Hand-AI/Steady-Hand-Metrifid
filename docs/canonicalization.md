# Canonicalization and execution identity

## Canonical semantic object domain

A canonical semantic object contains only `null`, booleans, exact integers, strict UTF-8 strings, arrays, string-keyed objects, tagged IEEE-754 binary64 objects, and normalized exact-rational objects. Python floats, bytes, bytearrays, tuples, sets, enum instances, dataclasses, paths, datetimes, `Decimal`, `Fraction`, and arbitrary mappings are not canonical primitives; an explicit value or schema adapter must convert them first.

Booleans never satisfy an integer field. Strings and object keys must encode as strict UTF-8; lone surrogates refuse. Schema-specific names additionally require nonempty strings and may reject U+0000.

## Tagged IEEE-754 binary64

The only canonical runtime-float representation is:

```json
{"kind":"ieee754_binary64","bits":"3ff0000000000000"}
```

`bits` is exactly 16 lowercase hexadecimal characters in network/big-endian display order. The encoding preserves finite values, positive and negative zero, positive and negative infinity, and every available NaN payload bit. The user string `"NaN"` remains an ordinary string and cannot collide with tagged numeric NaN.

`Binary64.to_exact_rational()` converts finite bits to their exact dyadic rational. Infinity and NaN refuse.

## Exact rational and decimal tokens

The primitive is exactly:

```json
{"numerator":1,"denominator":500}
```

Constructor and decoder rules are strict:

1. numerator and denominator are exact Python integers; booleans refuse;
2. denominator must be strictly greater than zero before reduction;
3. greatest-common-divisor reduction is mandatory;
4. zero is exactly `0/1`;
5. the sign is carried only by the numerator.

A negative or zero denominator is never normalized into the valid domain.

The accepted decimal-token grammar is:

```text
0
[1-9][0-9]*
0\.[0-9]+
[1-9][0-9]*\.[0-9]+
```

Whitespace, signs, leading zeros in a nonzero integer part, scientific notation, underscores, empty fractional digits, NaN, Infinity, and bare Python floats refuse.

## Strict JSON loading

`strict_json_loads` accepts UTF-8 `str` or `bytes` only. It rejects duplicate keys at every nesting depth, raw JSON floating-point tokens, NaN/Infinity extensions, invalid UTF-8, and malformed tagged values. It returns canonical primitive candidates; it does not instantiate public schema dataclasses implicitly.

## Canonical JSON bytes

The byte algorithm is fixed:

1. validate the complete canonical primitive domain;
2. sort object keys by Unicode code-point order;
3. encode UTF-8 without a byte-order mark;
4. use separators `,` and `:` with no discretionary whitespace;
5. disable NaN output;
6. append no trailing newline.

Raw floats are forbidden, so Python-version-specific float rendering cannot affect canonical bytes. Canonicalization does not normalize Unicode, fold case, trim strings, resolve paths, add timestamps, or call arbitrary `str()` methods.

## SHA-256 and self-hashes

`canonical_sha256(value)` is lowercase SHA-256 over `canonical_json_bytes(value)`. Every SHA field is exactly 64 lowercase hexadecimal characters.

For a self-hashed object:

1. remove the self-hash field entirely;
2. canonicalize the remaining object;
3. hash those bytes;
4. insert the digest;
5. validate by repeating the operation.

Replacing the field with `null` is not equivalent to removing it. A receipt or operational failure with a null owned self-hash is an in-memory candidate and is not serializable. Module-private unhashed primitive builders are used only by finalizers. Receipt finalization computes owned nested hashes before the outer `receipt_sha256`; operational failure finalization excludes only `failure_sha256`. Strict parsers accept completed wire objects only.

## Canonical comparison collections

Comparison limitations are always the complete `LimitationCode` registry tuple in registry order. Reason records use status-rule rank followed by frozen `ReasonCode` registry rank and the remaining frozen total ordering fields and identical duplicate records refuse. `reason_codes` is the exact stable first-occurrence projection of the ordered reason records. Available operational input digests are unique and emitted in `InputDigestCode` registry order.

These collections are decision-bearing and therefore participate in their enclosing self-hashes.

## Installed distribution identity

The installed-distribution identity is not the wheel archive hash and never falls back to a source-tree hash. It binds the code that is actually imported to one normal installed `metrifid` distribution.

Canonicalization, the exact-number helpers, and receipt parsing and validation are pure: they never
invoke the shared native runtime gate, so they work under Python 3.11 or newer without MuJoCo
admission, on Linux, on macOS, and on Windows. Commands that compile or step a model do require that
gate, and native Windows is unsupported for them; WSL is the documented route. Those commands use
the exact admitted package/native runtime and bind its identity to evidence. The resolver-selected
newest stable MuJoCo is the primary development and release profile—3.12.0 for the frozen 2026-08-22
snapshot—while exact retained older profiles remain compatibility-tested.

### Root and manifest binding

1. Resolve `importlib.metadata.distribution("metrifid")`.
2. Require a normal installed distribution with a valid `.dist-info/RECORD`; editable/source-only metadata is unsupported.
3. Parse `RECORD` directly and strictly. Every row must have a unique normalized POSIX path and a usable installed file target.
4. Determine one installation root from the distribution metadata.
5. Inspect every currently loaded `metrifid` module that has a file. Every such module must be a regular file under that same root and under the installed `metrifid/` package directory.
6. Require every loaded module file to appear in the distribution manifest.
7. Read the manifested member and loaded module bytes and require them to match at identity time.

Direct `RECORD` parsing is important: an API that silently omits a missing manifested file cannot be used as the completeness authority.

### Selected members

The canonical distribution manifest includes:

- every regular manifested file under `metrifid/`, excluding bytecode and `__pycache__`;
- `.dist-info/METADATA`;
- `.dist-info/WHEEL`;
- `.dist-info/entry_points.txt` when present;
- license files recorded inside the `.dist-info` directory.

It excludes `RECORD`, `INSTALLER`, `REQUESTED`, `direct_url.json`, bytecode, caches, and files outside the distribution.

For each selected member, record normalized POSIX relative path, byte length, and lowercase file SHA-256. Sort members by normalized path and hash this canonical object. The version shown below is a placeholder; a real identity uses the version read from the installed distribution metadata:

```json
{
  "schema":"metrifid.installed_distribution_identity",
  "schema_version":1,
  "distribution_name":"metrifid",
  "distribution_version":"<installed-version>",
  "members":[]
}
```

### Refusal boundary

No trusted distribution hash is returned when any of these conditions is observed:

- source-checkout execution;
- editable installation;
- `PYTHONPATH` or other module shadowing;
- mixed loaded `metrifid` module roots;
- loaded module absent from the manifest;
- loaded module or selected member missing, unreadable, non-regular, or changed;
- duplicate or unsafe normalized manifest path;
- malformed `RECORD`;
- installed metadata/version/root contradiction.

These are pre-contract execution-identity failures and are represented by `OperationalFailure` with an unverified tool observation and a `null` distribution hash.

## Collision boundaries

Tagged binary64 NaN cannot collide with a user string. Exact rational objects cannot collide with raw JSON floats because raw floats refuse. `true` and `1`, arrays and objects, field removal and `null`, operational failures and comparison receipts, verified and unbound tool identities, and distinct published documents remain distinguishable.

## Receipt decision-identity binding

The receipt embeds the complete `ComparisonContractIdentity`. Its canonical digest must match `inputs.comparison_contract_sha256`. Model-closure hashes, initial-state/action/alias semantic hashes, exact time rationals, monitored joints, alignment coverage, and the tolerance projection are cross-checked before a receipt can be completed or parsed. Recomputing only the outer receipt hash cannot legitimize a stale or contradictory inner identity.

## Operational identity-state binding

Operational self-hashes cover the structured reason, stage, and tool observation. Exact mismatch reasons require `MISMATCH`; exact unbound reasons require `UNBOUND`; post-identity stages require a verified installed-distribution hash. Quoted comparison tokens inside canonical evidence are inert strings and do not change the structured failure category.
