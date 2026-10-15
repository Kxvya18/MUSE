"""Fetch selected files from the public MSD Task01 tar with HTTP range requests
(walks the tar headers; downloads only the requested members)."""
import os, sys, urllib.request

URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"


def get(start, n):
    req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{start + n - 1}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main(out, wanted):
    wanted, offset = set(wanted), 0
    while wanted:
        h = get(offset, 512)
        if not h.strip(b"\0"):
            break
        name = h[:100].rstrip(b"\0").decode()
        prefix = h[345:500].rstrip(b"\0").decode()
        name = f"{prefix}/{name}" if prefix else name
        size = int(h[124:136].rstrip(b"\0 ").decode() or "0", 8)
        base = name.split("/")[-1]
        sub = name.split("/")[-2] if "/" in name else ""
        key = f"{sub}/{base}"
        dst = os.path.join(out, sub, base)
        if key in wanted and os.path.exists(dst) and os.path.getsize(dst) == size:
            wanted.discard(key)  # already fetched in an earlier attempt
        elif key in wanted:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            open(dst, "wb").write(get(offset + 512, size))
            wanted.discard(key)
            print("fetched", key, size, flush=True)
        offset += 512 + (size + 511) // 512 * 512
    print("missing:", sorted(wanted))


if __name__ == "__main__":
    cases = sys.argv[2:]
    main(sys.argv[1], [f"{s}/{c}.nii.gz" for c in cases for s in ("imagesTr", "labelsTr")])
