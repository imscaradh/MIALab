import datetime
import os
import sys
import timeit
import numpy as np
import SimpleITK as sitk
from sklearn import metrics
import matplotlib.pyplot as plt

import pymia.data.conversion as conversion
import pymia.evaluation.writer as writer

try:
    import mialab.data.structure as structure
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil
except ImportError:
    # Append the MIALab root directory to Python path
    sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[0]), '..'))
    import mialab.data.structure as structure
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil

LOADING_KEYS = [structure.BrainImageTypes.T1w,
                structure.BrainImageTypes.T2w,
                structure.BrainImageTypes.GroundTruth,
                structure.BrainImageTypes.BrainMask,
                structure.BrainImageTypes.RegistrationTransform]  # the list of data we will load


class ClassificationController():

    def __init__(self, classifiers: list, result_dir, data_atlas_dir, data_train_dir, data_test_dir):
        self.classifiers = [(clf, [], [], []) for clf in classifiers]

        for clf in self.classifiers:
            print(f'Classifier: {clf}')

        self.result_dir = result_dir
        self.data_atlas_dir = data_atlas_dir

        # load atlas images
        putil.load_atlas_images(data_atlas_dir)

        # crawl the training image directories
        crawler = futil.FileSystemDataCrawler(data_train_dir,
                                              LOADING_KEYS,
                                              futil.BrainImageFilePathGenerator(),
                                              futil.DataDirectoryFilter())

        pre_process_params = {'skullstrip_pre': True,
                              'normalization_pre': True,
                              'registration_pre': True,
                              'coordinates_feature': True,
                              'intensity_feature': True,
                              'gradient_intensity_feature': True}

        # load images for training and pre-process
        images = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=False)

        # generate feature matrix and label vector
        self.X_train = np.concatenate([img.feature_matrix[0] for img in images])
        self.y_train = np.concatenate([img.feature_matrix[1] for img in images]).squeeze()

        # crawl the test image directories
        crawler = futil.FileSystemDataCrawler(data_test_dir,
                                              LOADING_KEYS,
                                              futil.BrainImageFilePathGenerator(),
                                              futil.DataDirectoryFilter())

        # load images for testing and pre-process
        self.X_test = putil.pre_process_batch(crawler.data, {'training': False, **pre_process_params}, multi_process=False)

        # initialize evaluator
        self.evaluator = putil.init_evaluator()

    def train(self):
        for clf, _, _, _ in self.classifiers:
            print('-' * 5, 'Training...')

            start_time = timeit.default_timer()
            clf.fit(self.X_train, self.y_train)
            print(f' [{clf}] Time elapsed:', timeit.default_timer() - start_time, 's')

    def feature_importance(self):
        print('TODO: Implement feature importance')
        # TODO: Generalize for other classifiers
        # CURRENTLY ONLY FOR RFC
        # print the feature importance for the training
        # featureLabels = ["AtlasCoordsX", "AtlasCoordsY", "AtlasCoordsZ", "T1wIntensities", "T2wIntensities", "T1WGradient",
        #                 "T2wGradient"]
        # featureImportancesOrdered = (-clf.feature_importances_).argsort()
        # featureLabelsOrdered = [featureLabels[arg] for arg in featureImportancesOrdered]
        # featureImportancePrint = ["{}: {:.4f}".format(label, value) for label, value in
        #                         zip(featureLabelsOrdered, clf.feature_importances_[featureImportancesOrdered])]
        # print("Feature importance in descending order:\n", featureImportancePrint)

    def test(self):
        print('-' * 5, 'Testing...')

        # create a result directory with timestamp
        t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        result_dir = os.path.join(self.result_dir, t)
        os.makedirs(result_dir, exist_ok=True)

        for clf, y_true, y_pred, y_pred_proba in self.classifiers:
            for img in self.X_test:
                print('-' * 10, 'Testing', img.id_)

                start_time = timeit.default_timer()
                predictions = clf.predict(img.feature_matrix[0])
                probabilities = clf.predict_proba(img.feature_matrix[0])
                print(' Time elapsed:', timeit.default_timer() - start_time, 's')

                # convert prediction and probabilities back to SimpleITK images
                image_prediction = conversion.NumpySimpleITKImageBridge.convert(predictions.astype(np.uint8), img.image_properties)
                image_probabilities = conversion.NumpySimpleITKImageBridge.convert(probabilities, img.image_properties)

                # evaluate segmentation without post-processing
                self.evaluator.evaluate(image_prediction, img.images[structure.BrainImageTypes.GroundTruth], img.id_)

                y_true.append(sitk.GetArrayFromImage(img.images[structure.BrainImageTypes.GroundTruth]))
                y_pred.append(predictions)
                y_pred_proba.append(probabilities)

    def post_process(self):
        # post-process segmentation and evaluate with post-processing
        # post_process_params = {'simple_post': True}
        # images_post_processed = putil.post_process_batch(images_test, images_prediction, images_probabilities,
        #                                                  post_process_params, multi_process=True)
        pass

    def evaluate(self):
        for clf, y_true, y_pred, y_pred_proba in self.classifiers:
            y_true = np.concatenate(y_true, axis=0)
            y_pred = np.concatenate(y_pred, axis=0)
            fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)

            # # create ROC curve
            plt.ylabel('True Positive Rate (TPR)')
            plt.xlabel('False Positive Rate (FPR)')

            plt.plot(fpr, tpr, label=f'AUC={auc}')
            plt.legend(loc=4)

            plt.show()
            plt.savefig(os.path.join(self.result_dir, clf.__name__, 'roc.png'))

            #evaluation_results = self.evaluator.results
            # for i, img in enumerate(self.X_test):
            #     # save results
            #     sitk.WriteImage(y_pred[i], os.path.join(self.result_dir, self.X_test[i].id_ + '_SEG.mha'), True)

            # # use two writers to report the results
            # os.makedirs(self.result_dir, exist_ok=True)  # generate result directory, if it does not exists
            # result_file = os.path.join(self.result_dir, 'results.csv')
            # writer.CSVWriter(result_file).write(self.evaluator.results)

            # print('\nSubject-wise results...')
            # writer.ConsoleWriter().write(self.evaluator.results)

            # # report also mean and standard deviation among all subjects
            # result_summary_file = os.path.join(self.result_dir, 'results_summary.csv')
            # functions = {'MEAN': np.mean, 'STD': np.std}
            # writer.CSVStatisticsWriter(result_summary_file, functions=functions).write(self.evaluator.results)
            # print('\nAggregated statistic results...')
            # writer.ConsoleStatisticsWriter(functions=functions).write(self.evaluator.results)

            # # clear results such that the evaluator is ready for the next evaluation
            # self.evaluator.clear()
