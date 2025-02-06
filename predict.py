import json
import argparse
from deepfrier.Predictor import Predictor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--seq', type=str,  help="Protein sequence to be annotated.")
    parser.add_argument('-cm', '--cmap', type=str,  help="Protein contact map to be annotated (in *npz file format).")
    parser.add_argument('-pdb', '--pdb_fn', type=str,  help="Protein PDB file to be annotated.")
    parser.add_argument('--cmap_csv', type=str,  help="Catalogue with chain to file path mapping.")
    parser.add_argument('--pdb_dir', type=str,  help="Directory with PDB files of predicted Rosetta/DMPFold structures.")
    parser.add_argument('--fasta_fn', type=str,  help="Fasta file with protein sequences.")
    parser.add_argument('--model_config', type=str, default='./trained_models/model_config.json', help="JSON file with model names.")
    parser.add_argument('-ont', '--ontology', type=str, default=['mf'], nargs='+', required=True, choices=['mf', 'bp', 'cc', 'ec'],
                        help="Gene Ontology/Enzyme Commission.")
    parser.add_argument('-o', '--output_fn_prefix', type=str, default='DeepFRI', help="Save predictions/saliency in file.")
    parser.add_argument('-v', '--verbose', help="Prints predictions.", action="store_true")
    parser.add_argument('--use_guided_grads', help="Use guided grads to compute gradCAM.", action="store_true")
    parser.add_argument('--saliency', help="Compute saliency maps for every protein and every MF-GO term/EC number.", action="store_true")
    # New argument to supply a list of files to process (one file name per line) -Filip
    parser.add_argument('--file_list', type=str, help='File containing a list of PDB files to process')
    parser.add_argument('--mapping_file', type=str, help='CSV file mapping FASTA IDs to PDB files')
    args = parser.parse_args()

    with open(args.model_config) as json_file:
        params = json.load(json_file)

    
    if args.cmap is not None or args.pdb_fn is not None or args.cmap_csv is not None or args.pdb_dir is not None:
        params = params['gcn']
    elif args.seq is not None or args.fasta_fn is not None:
        params = params['cnn']

    gcn = params['gcn']
    layer_name = params['layer_name']
    models = params['models']

    for ont in args.ontology:
        predictor = Predictor(models[ont], gcn=gcn)

        if args.seq is not None and args.pdb_fn is not None:
            predictor.predict_from_pdb_and_seq(args.pdb_fn, args.seq)
        elif args.fasta_fn is not None and args.mapping_file is not None:
            predictor.predict_from_fasta_mapping(args.fasta_fn, args.mapping_file, pdb_dir=args.pdb_dir)
        elif args.seq is not None:
            predictor.predict(args.seq)
        elif args.cmap is not None:
            predictor.predict(args.cmap)
        elif args.pdb_fn is not None:
            predictor.predict(args.pdb_fn)
        elif args.fasta_fn is not None:
            predictor.predict_from_fasta(args.fasta_fn)
        elif args.cmap_csv is not None:
            predictor.predict_from_catalogue(args.cmap_csv)
        elif args.pdb_dir is not None and args.file_list is not None:
            predictor.predict_from_PDB_dir(args.pdb_dir, args.file_list)
        elif args.pdb_dir is not None:
            predictor.predict_from_PDB_dir(args.pdb_dir)

        # save predictions
        predictor.export_csv(args.output_fn_prefix + "_" + ont.upper() + "_predictions.csv", args.verbose)
        predictor.save_predictions(args.output_fn_prefix + "_" + ont.upper() + "_pred_scores.json")

        # save saliency maps
        if args.saliency and ont in ['mf', 'ec']:
            predictor.compute_GradCAM(layer_name=layer_name, use_guided_grads=args.use_guided_grads)
            predictor.save_GradCAM(args.output_fn_prefix + "_" + ont.upper() + "_saliency_maps.json")
