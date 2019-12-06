"""A module for exporting summaries of log databases.

Example usage:

  Print all of the parameters for a specific run:

    bazel run //deeplearning/ml4pl/models:export_logs -- \
      --log_db='sqlite:////tmp/logs.db' \
      --ascii_table=parameters \
      --run_id='ZeroR:191206111012:example'

  Export tables for all runs to a google sheets spreadsheet, with extra
  columns for the --initial_learning_rate and --learning_rate_decay flags:

    bazel run //deeplearning/ml4pl/models:export_logs -- \
      --log_db='sqlite:////tmp/logs.db' \
      --google_sheet='My Spreadsheet' \
      --google_sheets_credentials=/tmp/credentials.json \
      --google_sheets_default_share_with=joe@example.com \
      --extra_flags=initial_learning_rate,learning_rate_decay
"""
import pathlib
from typing import List

import pandas as pd

from deeplearning.ml4pl.models import log_database
from deeplearning.ml4pl.models import logger
from labm8.py import app
from labm8.py import google_sheets
from labm8.py import pdutil
from labm8.py import prof

# During table export, we be able to de-pickle any of the class objects from
# this project. Import the modules which define these classes here.

_unused_modules_ = (logger, log_database)


FLAGS = app.FLAGS

# Flags for filtering the results to export:
app.DEFINE_list(
  "run_id",
  [],
  "A list of run IDs to export. If not set, all runs in the database are "
  "exported. Run IDs that do not exist are silently ignored.",
)
# Flags for augmenting the data that is exported:
app.DEFINE_list(
  "extra_flags",
  [],
  "A list of additional flag parameters to add as columns to exported tables. "
  "E.g. --extra_flags=foo,bar creates two extra columns in exported tables "
  "containing the per-run values of --foo and --bar flags. By default, the "
  "--graph_db flag is exported. Flags that do not exist are silently ignored.",
)
# Flags that determine where to export tables to:
app.DEFINE_output_path(
  "csv_dir", None, "A directory to write CSV table files to."
)
app.DEFINE_list(
  "ascii_table",
  [],
  "A list of tables to print to stdout as ASCII-formatted tables. Valid table "
  "names: {parameters,epochs,runs}.",
)
app.DEFINE_string(
  "google_sheet",
  None,
  "The name of a Google Sheets spreadsheet to export tables to. If it does "
  "not exist, the spreadsheet is created and shared with "
  "--google_sheets_default_share_with. See --google_sheets_credentials for "
  "setting the credentials required to use the Google Sheets API.",
)


class TableExporter(object):
  """An object that exports tables."""

  def OnTable(self, name: str, df: pd.DataFrame):
    """Export a table with the given name."""
    raise NotImplementedError("abstract class")


class CsvExport(TableExporter):
  """Write CSV files of tables."""

  def __init__(self, outdir: pathlib.Path):
    self.outdir = outdir
    self.outdir.mkdir(exist_ok=True, parents=True)

  def OnTable(self, name: str, df: pd.DataFrame):
    outpath = self.outdir / f"{name}.csv"
    with prof.Profile(f"Wrote CSV {outpath}"):
      df.to_csv(outpath, index=False)


class GoogleSheetsExport(TableExporter):
  """Export tables to a spreadsheet."""

  def __init__(self, spreadsheet_name: str):
    self.spreadsheet_name = spreadsheet_name
    self.google = google_sheets.GoogleSheets.FromFlagsOrDie()
    self.spreadsheet = self.google.GetOrCreateSpreadsheet(self.spreadsheet_name)

  def OnTable(self, name: str, df: pd.DataFrame):
    with prof.Profile(f"Exported worksheet {self.spreadsheet_name}:{name}"):
      worksheet = self.google.GetOrCreateWorksheet(self.spreadsheet, name)
      self.google.ExportDataFrame(worksheet, df, index=False)


class AsciiTableExport(TableExporter):
  """Print tables to stdout."""

  def __init__(self, names: List[str]):
    self.names = names

  def OnTable(self, name: str, df: pd.DataFrame):
    if name in self.names:
      print(pdutil.FormatDataFrameAsAsciiTable(df))


def Main():
  """Main entry point."""
  log_db = FLAGS.log_db()

  # Create the exporters objects from flags.
  exporters = []
  if FLAGS.csv_dir:
    exporters.append(CsvExport(FLAGS.csv_dir))
  if FLAGS.google_sheet:
    exporters.append(GoogleSheetsExport(FLAGS.google_sheet))
  if FLAGS.ascii_table:
    exporters.append(AsciiTableExport(FLAGS.ascii_table))
  if not exporters:
    raise app.UsageError("No exporters")

  for name, df in log_db.GetTables(FLAGS.run_id, extra_flags=FLAGS.extra_flags):
    for exporter in exporters:
      exporter.OnTable(name, df)


if __name__ == "__main__":
  app.Run(Main)