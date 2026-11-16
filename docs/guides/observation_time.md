# Observation selection for publications

This minimal C04 module belongs to `MAP-OBSERVATION-TIME`. It does not close
that ticket or qualify general temporal authoring, frame replay or interpolation.

Import `ObservationRecord` and `ObservationTrack` from
`forge3d.observation_time`. Records declare an acquisition time, a half-open
validity interval `[valid_from, valid_until)` and a JSON payload. Times must be
timezone-aware ISO strings or datetimes. Comparisons and elapsed age normalize
to UTC; captions retain each declared timezone and express elapsed age in days.
Tables retain both UTC timestamps and the declared-offset timestamps. Records
are unavailable before acquisition or the start of their validity interval.

An explicit `no-data` policy leaves gaps missing. An explicit `hold` policy
uses the most recently acquired available record when no interval is active.
Overlapping active records select the latest acquisition. Equal acquisition
times among the newest candidates are ambiguous and raise instead of silently
choosing an edition. No values are interpolated.

`track.select(presentation_times)` returns the selected record, state and age
for each declared time. Its caption comes from that same selection and visibly
distinguishes missing and held observations. `track.selection_table(times)`
returns the policy, original records and selections as JSON-compatible data
for retaining with outputs. Record payloads are snapshotted at construction;
later edits to the input mapping cannot relabel a retained selection.

C06 consumes these selections and their captions. It must not independently
derive observation years or substitute datasets for missing observations.
The C04 record exposes `observation_year` in its declared timezone; C06 rejects
a source year that disagrees with it before writing time-series output.
