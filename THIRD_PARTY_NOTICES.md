# Third-party notices

## MS-TCN++

- Source: https://github.com/sj-li/MS-TCN2
- Pinned source commit: `9d31fb3c23467b9ce3030d43b6d33a96869b6422`
- Bundled file: `stagefusion/model.py`, copied without changes.
- License: [MIT](LICENSES/MS-TCN2-MIT.txt), copyright (c) 2019 yabufarha.
- The newest upstream `model.py` was not used because it contains a syntax error. The pinned earlier revision supplies the same `MS_TCN2` constructor required by the provided source.

This dependency completes a missing import in the source snapshot. It has not been represented as a recovered copy of the authors' original server file.

SAM2, Ultralytics, timm, PyTorch and the other dependencies remain separate upstream projects under their own terms. Their model weights are not redistributed.

The authors have not specified a project-wide license for their original source. This release preserves third-party notices without applying their license to unrelated author code.
