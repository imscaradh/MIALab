import datetime
import os
import sys
import timeit

import matplotlib.pyplot as plt
import numpy as np
import pymia.data.conversion as conversion
import pymia.evaluation.writer as writer
import SimpleITK as sitk
from sklearn import metrics, preprocessing
from sklearn.inspection import permutation_importance

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

    def __init__(self, classifiers: list, result_dir, data_atlas_dir, data_train_dir, data_test_dir, limit=0):
        self.classifiers = [(clf, []) for clf in classifiers]

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

        if limit > 0:
            crawler.data = dict(list(crawler.data.items())[:limit])

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

        if limit > 0:
            crawler.data = dict(list(crawler.data.items())[:limit])

        # load images for testing and pre-process
        images = putil.pre_process_batch(crawler.data, {'training': False, **pre_process_params}, multi_process=False)
        self.X_test = np.concatenate([img.feature_matrix[0] for img in images])
        self.y_true = np.concatenate([img.images[structure.BrainImageTypes.GroundTruth] for img in images])  # WTF

        # initialize evaluator
        self.evaluator = putil.init_evaluator()

    def train(self):
        for clf, _ in self.classifiers:
            print('-' * 5, f'Training for {clf}...')

            start_time = timeit.default_timer()
            clf.fit(self.X_train, self.y_train)
            print(f' Time elapsed:', timeit.default_timer() - start_time, 's')

    def feature_importance(self):
        # get feature matrix for test images
        data_test = np.concatenate([img.feature_matrix[0] for img in self.X_test])
        # get ground truth for test images
        data_labels = preprocessing.label_binarize(
            np.concatenate([img.feature_matrix[1] for img in self.X_test]).squeeze(),
            classes=[1, 2, 3, 4, 5]
        )
        # for better readability replace features (ints) with strings
        feature_labels = ["AtlasCoordsX", "AtlasCoordsY", "AtlasCoordsZ", "T1wIntensities", "T2wIntensities", "T1WGradient", "T2wGradient"]

        for clf, _ in self.classifiers:
            result = permutation_importance(clf, data_test, data_labels, random_state=42, scoring='accuracy')
            importance_order = (-result.importances_mean).argsort()
            labels_odered = [feature_labels[arg] for arg in importance_order]
            means_odered = result.importances_mean[importance_order]
            sd_ordered = result.importances_std[importance_order]

            # print out at the moment, but change so that is stored in a csv...
            printMe = ["{}: mean: {:.4f}, sd: {:.4f}".format(label, mean, sd) for label, mean, sd in
                       zip(labels_odered, means_odered, sd_ordered)]
            print("Feature importance in descending order:\n", printMe)

    def test(self):
        for clf, y_pred in self.classifiers:
            print('-' * 5, f'Testing with {clf}...')
            y_pred.append(clf.predict(self.X_test))  # TODO: Move away from list structure

            # for img in self.X_test:
            #     print('-' * 10, 'Testing', img.id_)

            #     start_time = timeit.default_timer()
            #     predictions = clf.predict(img.feature_matrix[0])
            #     probabilities = clf.predict_proba(img.feature_matrix[0])

            #     print(f'{" " * 10} Time for prediction elapsed: {timeit.default_timer() - start_time:.2f}s')

            #     image_prediction = conversion.NumpySimpleITKImageBridge.convert(predictions.astype(np.uint8), img.image_properties)
            #     image_reference = img.images[structure.BrainImageTypes.GroundTruth]
            #     # image_probabilities = conversion.NumpySimpleITKImageBridge.convert(probabilities, img.image_properties)

            #     prediction_array = sitk.GetArrayFromImage(image_prediction)
            #     reference_array = sitk.GetArrayFromImage(image_reference)
            #     # probabilities_array = sitk.GetArrayFromImage(image_probabilities)

            #     y_true.append(reference_array)
            #     y_pred.append(prediction_array)
            # y_pred_proba.append(probabilities_array)

            # evaluate segmentation without post-processing
            # self.evaluator.evaluate(image_prediction, img.images[structure.BrainImageTypes.GroundTruth], img.id_)

    def post_process(self):
        # post-process segmentation and evaluate with post-processing
        # images_post_processed = putil.post_process_batch(images_test, images_prediction, images_probabilities, {'simple_post': True}, multi_process=True)
        pass

    def evaluate(self):
        for clf, y_pred in self.classifiers:

            for label, label_str in putil.labels.items():
                fpr1, tpr1, _ = metrics.roc_curve(self.y_true, np.concatenate(y_pred, axis=0), pos_label=label)
                plt.plot(fpr1, tpr1, label=label_str)

            # y_true_label = np.array([])
            # y_pred_label = np.array([])

            # for label, label_str in putil.labels.items():
            #     reference_of_label = np.in1d(reference_array.ravel(), label, True).reshape(reference_array.shape).astype(np.uint8)
            #     prediction_of_label = np.in1d(prediction_array.ravel(), label, True).reshape(prediction_array.shape).astype(np.uint8)

            #     for reference_array in y_true:
            #         y_true_label = np.concatenate((y_true_label, np.in1d(reference_array.ravel(), label, True).flatten()), axis=0)
            #     for prediction_array in y_pred:
            #         y_pred_label = np.concatenate((y_pred_label, np.in1d(prediction_array.ravel(), label, True).flatten()), axis=0)

            #     fpr, tpr, _ = metrics.roc_curve(y_true_label, y_pred_label)

            #     # Create ROC curve
            #     plt.plot(fpr, tpr, label=label_str)

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
