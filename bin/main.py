"""A medical image analysis pipeline.

The pipeline is used for brain tissue segmentation using a decision forest classifier.
"""
import argparse
import os
import sys

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import sklearn.ensemble as sk_ensemble
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

try:
    from mialab.classifier.classifier_controller import ClassificationController
except ImportError:
    # Append the MIALab root directory to Python path
    sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[0]), '..'))
    from mialab.classifier.classifier_controller import ClassificationController


def main(result_dir: str, data_atlas_dir: str, data_train_dir: str, data_test_dir: str):
    """Brain tissue segmentation using decision forests.

    The main routine executes the medical image analysis pipeline:

        - Image loading
        - Registration
        - Pre-processing
        - Feature extraction
        - Decision forest classifier model building
        - Segmentation using the decision forest classifier model on unseen images
        - Post-processing of the segmentation
        - Evaluation of the segmentation
    """
    # parameters for grid search
    params_rfc = {'n_estimators': [5, 15], 'max_features': [5,15], 'max_depth': [5,15]}
    params_knn = {'n_neighbors': [1,10], 'weights': ('uniform', 'distance')}
    params_svc = {'kernel': ('linear', 'rbf', 'poly', 'sigmoid'), 'C': [1,10],'gamma' : ('auto', 'scale')}

    params = [
        {'n_neighbors': [1, 10], 'weights': ('uniform', 'distance')},
        {'n_estimators': [5, 15], 'max_features': [2,5,7], 'max_depth': [5, 15]}

    ]
    rfc = GridSearchCV(sk_ensemble.RandomForestClassifier, params_rfc)
    knn = GridSearchCV(KNeighborsClassifier, params_knn)
    svc = GridSearchCV(SVC, params_svc)

    # replace classifiers with variables above -> does not work yet :
    cc = ClassificationController([
        KNeighborsClassifier(n_neighbors=1, weights='distance'),
        sk_ensemble.RandomForestClassifier(max_features=7, n_estimators=10,max_depth=10)
    ], result_dir, data_atlas_dir, data_train_dir, data_test_dir, params, limit=1)


    cc.train()
    # cc.feature_importance()
    cc.test()
    cc.post_process()
    cc.evaluate()


if __name__ == "__main__":
    """The program's entry point."""
    np.random.seed(42)

    script_dir = os.path.dirname(sys.argv[0])

    parser = argparse.ArgumentParser(description='Medical image analysis pipeline for brain tissue segmentation')

    parser.add_argument(
        '--result_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, './mia-result')),
        help='Directory for results.'
    )

    parser.add_argument(
        '--data_atlas_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/atlas')),
        help='Directory with atlas data.'
    )

    parser.add_argument(
        '--data_train_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/train/')),
        help='Directory with training data.'
    )

    parser.add_argument(
        '--data_test_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/test/')),
        help='Directory with testing data.'
    )

    args = parser.parse_args()
    main(args.result_dir, args.data_atlas_dir, args.data_train_dir, args.data_test_dir)
