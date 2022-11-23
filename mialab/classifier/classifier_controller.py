import datetime
import os
import sys
import timeit
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pymia.data.conversion as conversion
import pymia.evaluation.writer as writer
import SimpleITK as sitk
from sklearn import metrics, preprocessing
from sklearn.inspection import permutation_importance
import csv

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

    def __init__(self, classifiers: list, result_dir, data_atlas_dir, data_train_dir, data_test_dir, limit=0, preload_data=True):
        self.classifiers = [(clf, [], []) for clf in classifiers]

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

        if limit > 0:
            crawler.data = dict(list(crawler.data.items())[:limit])

        # load images for training and pre-process
        def data_train_loader(): return putil.pre_process_batch(crawler.data, pre_process_params, multi_process=False)

        if preload_data:
            images = self._preload_data('train_preprocessed.pyo', data_train_loader)
        else:
            images = data_train_loader()

        # generate feature matrix and label vector
        self.X_train = np.concatenate([img.feature_matrix[0] for img in images])
        self.y_train = np.concatenate([img.feature_matrix[1] for img in images]).squeeze()
        
        # crawl the test image directories
        crawler = futil.FileSystemDataCrawler(data_test_dir,
                                              LOADING_KEYS,
                                              futil.BrainImageFilePathGenerator(),
                                              futil.DataDirectoryFilter())

        if limit > 0:
            crawler.data = dict(list(crawler.data.items())[:limit])

        # load images for testing and pre-process
        def data_test_loader(): return putil.pre_process_batch(crawler.data, {'training': False, **pre_process_params}, multi_process=False)

        if preload_data:
            self.X_test = self._preload_data('test_preprocessed.pyo', data_test_loader)
        else:
            self.X_test = data_test_loader()

        self.y_true = np.concatenate([img.images[structure.BrainImageTypes.GroundTruth] for img in images])  # WTF

        # initialize evaluator
        # self.evaluator = putil.init_evaluator()

    def _preload_data(self, file_name, data_loader):
        if not os.path.exists(file_name):
            print(f'File {file_name} does not exist, dumping...')
            file = open(file_name, 'wb')
            pickle.dump(data_loader(), file)
            file.close()
        else:
            print(f'File {file_name} found, loading...')

        file = open(file_name, 'rb')
        data = pickle.load(file)
        file.close()
        return data
        

    def train(self):
        for clf, _, _ in self.classifiers:
            print('-' * 5, f'Training for {clf.__class__.__name__}...')

            start_time = timeit.default_timer()
            clf.fit(self.X_train, self.y_train)
            print(f' Time elapsed: {timeit.default_timer() - start_time:.2f}s')

    def feature_importance(self):
        # get feature matrix for test images
        data_test = np.concatenate([img.feature_matrix[0] for img in self.X_test])
        # get ground truth for test images
        data_labels = preprocessing.label_binarize(
            np.concatenate([img.feature_matrix[1] for img in self.X_test]).squeeze(),
            classes=[1, 2, 3, 4, 5]
        )
        header = ["classifier", "atlas_x", "atlas_y", "atlas_z", "T1w_intensity", "T2w_intensity",
                  "T1w_gradient", "T2w_gradient"]

        # prepare csv to store results
        os.makedirs(self.result_dir, exist_ok=True)  # generate result directory, if it does not exists
        with open(os.path.join(self.result_dir, 'feature_importances.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for clf, _, _ in self.classifiers:
                result = permutation_importance(clf, data_test, data_labels, random_state=42, scoring='accuracy')

                means_list = result.importances_mean.tolist()
                # add classifier configuration as first column
                means_list.insert(0, clf)
                writer.writerow(means_list)

                sd_list = result.importances_std.tolist()
                # add classfier configuration as first column
                sd_list.insert(0, clf)
                writer.writerow(sd_list)

    def test(self):
        for clf, y_pred, y_pred_proba in self.classifiers:
            print('-' * 5, f'Testing with {clf.__class__.__name__}...')

            for img in self.X_test:
                print('-' * 10, 'Testing', img.id_)

                start_time = timeit.default_timer()
                predictions = clf.predict(img.feature_matrix[0])
                probabilities = clf.predict_proba(img.feature_matrix[0])

                print(f'{" " * 10} Time for prediction elapsed: {timeit.default_timer() - start_time:.2f}s')

                image_prediction = conversion.NumpySimpleITKImageBridge.convert(predictions.astype(np.uint8), img.image_properties)
                image_probabilities = conversion.NumpySimpleITKImageBridge.convert(probabilities, img.image_properties)

                prediction_array = sitk.GetArrayFromImage(image_prediction)
                probabilities_array = sitk.GetArrayFromImage(image_probabilities)

                y_pred.append(prediction_array)
                y_pred_proba.append(probabilities_array)

            # evaluate segmentation without post-processing
            # self.evaluator.evaluate(image_prediction, img.images[structure.BrainImageTypes.GroundTruth], img.id_)

    def post_process(self):
        """ This is not part of our project """
        pass

    def evaluate(self):
        for clf, y_pred, _ in self.classifiers:
            y_pred = np.concatenate(y_pred, axis=0).flatten()

            for label, label_str in putil.labels.items():
                # ROC
                fpr, tpr, _ = metrics.roc_curve(self.y_true, y_pred, pos_label=label)
                plt.plot(fpr, tpr, label=label_str)

                # AUC
                auc = metrics.auc(fpr, tpr)
                print(f'AUC for label {label_str}: {auc:.2f}')
                

            plt.ylabel('True Positive Rate (TPR)')
            plt.xlabel('False Positive Rate (FPR)')
            plt.legend(loc=4)

            # create a result directory with timestamp
            t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            output_dir = os.path.join(self.result_dir, t)
            os.makedirs(output_dir, exist_ok=True)

            plt.savefig(os.path.join(output_dir, f'{clf.__class__.__name__.lower()}_roc.png'))

            # evaluation_results = self.evaluator.results
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
