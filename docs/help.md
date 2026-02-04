# Help

## Troubleshooting

- **Installation fails**: Use a virtual environment and verify Python 3.10+.
- **Dependency conflicts**: Install Trackers in a clean environment.
- **Eval file not found**: Check your path and file extensions (`.txt`).
- **No sequences found**: Verify your directory layout or provide a `seqmap`.

## FAQ

**Why are metrics printed as percentages?**
Table output matches TrackEval formatting by printing float metrics as percentages with 3 decimal places.

**Do tracker outputs need to match ground truth format?**
Yes. Both ground truth and tracker predictions must be MOT Challenge text files.

**What is the difference between CLEAR, HOTA, and Identity metrics?**
CLEAR focuses on detection errors and ID switches, HOTA balances detection and association, and Identity focuses on global ID consistency.

**Can I evaluate multiple trackers?**
Yes. Run separate evaluations or override `tracker_name` when using MOT17 layouts.

## Support and community

- GitHub issues: https://github.com/roboflow/trackers/issues
- Discord: https://discord.gg/GbfgXGJ8Bk

## Contributing

- Contributing guide: https://github.com/roboflow/trackers/blob/main/CONTRIBUTING.md
- Code of Conduct: https://github.com/roboflow/trackers/blob/main/CODE_OF_CONDUCT.md
