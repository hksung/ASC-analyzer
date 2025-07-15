#!/usr/bin/env python
"""
Command-line interface for the ASC Analyzer.
By default, it processes all .txt files under data/text and writes
l2Written_{SOURCE}.csv under data/. Users can override input/output via args,
choose reference 'cow' or 'subt' data, and select exactly which indices to output.
"""
import argparse
import json
import glob
import os

from .core import indexCalcFull, writeCsv, processCorpusASC, ascExtractDoc

# -----------------------------------------------------------------------------
# DEFAULT INDICES
# Edit this list to change which columns are written to the CSV by default.
DEFAULT_INDICES = [
    'clauseCount',
    'clauseCountNoBe',
    'ascMATTR11',
    'ascLemmaMATTR11',
    'ascLemmaMATTR11NoBe',
    'TRAN-S_Prop',
    'ATTR-Prop',
    'INTRAN-S_Prop',
    'PASSIVE_Prop',
    'INTRAN-MOT_Prop',
    'TRAN-RES_Prop',
    'CAUS-MOT_Prop',
    'DITRAN_Prop',
    'INTRAN-RES_Prop',
    'ascAvFreq',
    'ascLemmaAvFreq',
    'ascAvMI',
    'ascAvTscore',
    'ascAvDPLemmaCue',
    'ascAvDPStructureCue',
    'TRAN-S_AvMI',
    'INTRAN-S_AvMI',
    'PASSIVE_AvMI',
    'INTRAN-MOT_AvMI',
    'TRAN-RES_AvMI',
    'CAUS-MOT_AvMI',
    'DITRAN_AvMI',
    'INTRAN-RES_AvMI',
    'TRAN-S_Tscore',
    'INTRAN-S_Tscore',
    'PASSIVE_Tscore',
    'INTRAN-MOT_Tscore',
    'TRAN-RES_Tscore',
    'CAUS-MOT_Tscore',
    'DITRAN_Tscore',
    'INTRAN-RES_Tscore',
    'TRAN-S_DPLemmaCue',
    'INTRAN-S_DPLemmaCue',
    'PASSIVE_DPLemmaCue',
    'INTRAN-MOT_DPLemmaCue',
    'TRAN-RES_DPLemmaCue',
    'CAUS-MOT_DPLemmaCue',
    'DITRAN_DPLemmaCue',
    'INTRAN-RES_DPLemmaCue',
    'TRAN-S_DPStructureCue',
    'INTRAN-S_DPStructureCue',
    'PASSIVE_DPStructureCue',
    'INTRAN-MOT_DPStructureCue',
    'TRAN-RES_DPStructureCue',
    'CAUS-MOT_DPStructureCue',
    'DITRAN_DPStructureCue',
    'INTRAN-RES_DPStructureCue',
]
# -----------------------------------------------------------------------------

def main():
    # Get the path to the current file (asc_analyzer/cli.py)
    pkg_dir = os.path.abspath(os.path.dirname(__file__))
    default_db = os.path.join(pkg_dir, 'data')

    parser = argparse.ArgumentParser(
        prog='asc-analyzer',
        description='Compute ASC-based indices for a directory of text files',
        allow_abbrev=False
    )
    parser.add_argument(
        '--input-dir',
        default=os.path.join(default_db, 'test'),
        help='Directory containing .txt files to process (default: data/text)'
    )
    parser.add_argument(
        '--output-csv',
        default=None,
        help='Path to save the resulting CSV (default: data/Written_<source>.csv)'
    )
    parser.add_argument(
        '--source',
        choices=['cow', 'subt'],
        default='cow',
        help="Reference dataset to use: 'cow' (default) or 'subt'"
    )
    parser.add_argument(
        '--indices',
        help=(
            "Comma-separated list of the index names to include in the CSV. "
            "If omitted, uses DEFAULT_INDICES defined at the top of this file."
        )
    )
    parser.add_argument(
        '--save-asc-output',
        action='store_true',
        help='Save ASC-tagged outputs for each file (as *_ASCinfo.txt in input dir)'
    )

    parser.add_argument(
        '--print-asc',
        action='store_true',
        help='Print raw ASC tagging results to stdout (token + ASC label)'
    )

    args = parser.parse_args()

    # Determine filenames based on chosen source
    if args.source == 'cow':
        freq_fname     = 'allFreqsCOW_Refined_20240315.json'
        asc_soa_fname  = 'ascSoaCOW_20240305.json'
        default_output = os.path.join(default_db, 'Written_COW.csv')
    else:
        freq_fname     = 'allFreqsSubT_20240305.json'
        asc_soa_fname  = 'ascSoaSubT_20240305.json'
        default_output = os.path.join(default_db, 'Spoken_SubT.csv')

    # Resolve output path
    output_csv = args.output_csv or default_output

    # Paths for JSONs
    freq_path    = os.path.join(default_db, freq_fname)
    asc_soa_path = os.path.join(default_db, asc_soa_fname)

    # Validate JSONs
    for p in (freq_path, asc_soa_path):
        if not os.path.isfile(p):
            parser.error(f'Required file not found: {p}')

    # Load JSON data
    with open(freq_path,    errors='ignore') as f:
        freqD = json.load(f)
    with open(asc_soa_path, errors='ignore') as f:
        ascD = json.load(f)

    # Collect text files
    txt_files = glob.glob(os.path.join(args.input_dir, '*.txt'))
    if not txt_files:
        parser.error(f'No .txt files found in {args.input_dir}')

    # Compute indices
    results = indexCalcFull(txt_files, freqD, ascD)

    # Optional: Save ASC-tagged files or print them to stdout
    if args.save_asc_output or args.print_asc:
        for fname in txt_files:
            name = os.path.basename(fname)
            txt = open(fname, errors='ignore').read()
            ascList = ascExtractDoc(txt, freqD, ascD)

            # Build a minimal version with only 4 columns: idx, token, lemma, asc
            minimalAscList = []
            for s_idx, sent in enumerate(ascList):
                new_sent = [f"# sent_id = {s_idx + 1}"]  # 1-based indexing
                for row in sent:
                    trimmed = row[:4] + [""] * (4 - len(row))  # ensure it has at least 4 elements
                    idx = str(trimmed[0])
                    token = trimmed[1]
                    lemma = trimmed[2]
                    asc_raw = trimmed[3]
                    asc = "" if "\t" in asc_raw else (asc_raw if asc_raw not in ["_", ""] else "")
                    new_sent.append([idx, token, lemma, asc])
                minimalAscList.append(new_sent)

            # Save to file
            if args.save_asc_output:
                outfname = os.path.join(args.input_dir, name.replace('.txt', '_ASCinfo.txt'))
                with open(outfname, 'w') as outf:
                    for sent in minimalAscList:
                        for row in sent:
                            if isinstance(row, list):
                                outf.write('\t'.join(row) + '\n')
                            else:
                                outf.write(row + '\n')  # comment line
                        outf.write('\n')

            # Print to stdout
            if args.print_asc:
                print(f"\n# filename = {name}")
                for sent in minimalAscList:
                    for row in sent:
                        if isinstance(row, list):
                            print("\t".join(row))
                        else:
                            print(row)
                    print()



    # Determine which columns to write
    all_indices = list(next(iter(results.values())).keys())
    if args.indices:
        requested = [i.strip() for i in args.indices.split(',')]
        missing = set(requested) - set(all_indices)
        if missing:
            parser.error(f"Unknown indices requested: {', '.join(missing)}")
        index_names = requested
    else:
        # Filter DEFAULT_INDICES to only those actually present
        index_names = [i for i in DEFAULT_INDICES if i in all_indices]

    # Write CSV
    writeCsv(results, index_names, output_csv)
    print(f'Wrote {len(results)} rows to {output_csv}')


if __name__ == '__main__':
    main()
