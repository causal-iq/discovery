#
# Run evaluation of noisy data files
#

from argparse import ArgumentParser  # Standard command line arg parser

from fileio.noisy import evaluate_noisy

if __name__ == '__main__':
    argParser = ArgumentParser()
    argParser.add_argument("-s", "--strict", type=bool, default=False,
                           const=True, nargs="?",
                           help="Whether to apply strict validation")
    argParser.add_argument("-w", "--warnings", type=bool, default=False,
                           const=True, nargs="?",
                           help="Whether to report problems with graphs")
    argParser.add_argument("-k", "--ken", default=False,
                           const=True, nargs="?",
                           help="Whether to scan ken's local files")
    argParser.add_argument("-a", "--algorithm",
                           help="Only process graphs by these algorithms")
    argParser.add_argument("-n", "--network",
                           help="Only process graphs for these networks")
    argParser.add_argument("-no", "--noise",
                           help="Only process graphs for these noise cases")
    argParser.add_argument("-si", "--size",
                           help="Only process graphs for these data sizes")
    args = argParser.parse_args()

    filter = {}
    if args.algorithm:
        filter['algorithm'] = args.algorithm.split(',')
    if args.network:
        filter['network'] = args.network.split(',')
    if args.noise:
        filter['noise'] = args.noise.split(',')
    if args.size:
        filter['size'] = args.size.split(',')

    print('Scanning {}\'s graphs with {}strict validation with filter\n{} ...'
          .format(('Ken' if args.ken else 'team'),
                  ('' if args.strict else 'non-'), filter))

    LEARNED_DIR = 'c:/dev/noisy/dropbox-copy/Graphs learned' if args.ken \
        else 'c:/Users/ken/Dropbox/Noisy-data-paper/Graphs learned'
    TRUE_DIR = 'c:/dev/noisy/dropbox-copy/Graphs true'
    RESULTS_DIR = 'c:/dev/noisy/dropbox-copy/Results'

    evaluate_noisy(LEARNED_DIR, TRUE_DIR, RESULTS_DIR, strict=args.strict,
                   warnings=args.warnings, filter=filter)
