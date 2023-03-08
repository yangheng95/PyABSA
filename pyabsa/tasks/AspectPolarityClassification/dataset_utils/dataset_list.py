# -*- coding: utf-8 -*-
# file: dataset_list.py
# time: 02/11/2022 19:35
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.

from pyabsa.utils.data_utils.dataset_item import DatasetItem


class APCDatasetList(list):
    """
    The following datasets are for aspect polarity classification task.
    The datasets are collected from different sources, you can use the id to locate the dataset.
    """

    Laptop14 = DatasetItem("Laptop14", "113.Laptop14")
    Restaurant14 = DatasetItem("Restaurant14", "114.Restaurant14")

    # https://github.com/zhijing-jin/ARTS_TestSet
    ARTS_Laptop14 = DatasetItem("ARTS_Laptop14", "111.ARTS_Laptop14")
    ARTS_Restaurant14 = DatasetItem("ARTS_Restaurant14", "112.ARTS_Restaurant14")

    Restaurant15 = DatasetItem("Restaurant15", "115.Restaurant15")
    Restaurant16 = DatasetItem("Restaurant16", "116.Restaurant16")

    # Twitter
    ACL_Twitter = DatasetItem("Twitter", "101.ACL_Twitter")

    MAMS = DatasetItem("MAMS", "109.MAMS")

    # @R Mukherjee et al.
    Television = DatasetItem("Television", "117.Television")
    TShirt = DatasetItem("TShirt", "118.TShirt")

    # @WeiLi9811 https://github.com/WeiLi9811
    Yelp = DatasetItem("Yelp", "119.Yelp")

    # Chinese (binary polarity)
    Phone = DatasetItem("Phone", "107.Phone")
    Car = DatasetItem("Car", "104.Car")
    Notebook = DatasetItem("Notebook", "106.Notebook")
    Camera = DatasetItem("Camera", "103.Camera")

    # Chinese (triple polarity)
    # brightgems@github https://github.com/brightgems
    # Note that the annotation strategy of this dataset is highly different from other datasets,
    # please dont mix this dataset with any other dataset in trainer
    Shampoo = DatasetItem("Shampoo", "108.Shampoo")
    # jmc123@github https://github.com/jmc-123
    MOOC = DatasetItem("MOOC", "105.MOOC")
    MOOC_En = DatasetItem("MOOC_En", "121.MOOC_En")

    # https://www.kaggle.com/datasets/cf7394cb629b099cf94f3c3ba87e1d37da7bfb173926206247cd651db7a8da07
    Kaggle = DatasetItem("Kaggle", "129.Kaggle")

    Chinese_Zhang = DatasetItem("Chinese_Zhang", ["130.Chinese_Zhang"])

    # assembled dataset
    Chinese = DatasetItem(
        "Chinese",
        [
            "107.Phone",
            "103.Camera",
            "106.Notebook",
            "104.Car",
            "105.MOOC",
            "130.Chinese_Zhang",
        ],
    )
    Binary_Polarity_Chinese = DatasetItem(
        "Chinese", ["107.Phone", "103.Camera", "106.Notebook", "104.Car"]
    )
    Triple_Polarity_Chinese = DatasetItem("Chinese3way", ["105.MOOC"])

    SemEval2016Task5 = DatasetItem("SemEval2016Task5", ["120.SemEval2016Task5"])
    Arabic_SemEval2016Task5 = DatasetItem("Arabic_SemEval2016Task5", ["122.Arabic"])
    Dutch_SemEval2016Task5 = DatasetItem("Dutch_SemEval2016Task5", ["123.Dutch"])
    Spanish_SemEval2016Task5 = DatasetItem("Spanish_SemEval2016Task5", ["127.Spanish"])
    Turkish_SemEval2016Task5 = DatasetItem("Turkish_SemEval2016Task5", ["128.Turkish"])
    Russian_SemEval2016Task5 = DatasetItem("Russian_SemEval2016Task5", ["126.Russian"])
    French_SemEval2016Task5 = DatasetItem("French_SemEval2016Task5", ["125.French"])
    English_SemEval2016Task5 = DatasetItem("English_SemEval2016Task5", ["124.English"])

    English = DatasetItem(
        "English",
        [
            "113.Laptop14",
            "114.Restaurant14",
            "116.Restaurant16",
            "101.ACL_Twitter",
            "109.MAMS",
            "117.Television",
            "118.TShirt",
            "119.Yelp",
            "121.MOOC_En",
            "129.Kaggle",
        ],
    )

    # Abandon rest15 dataset due to data leakage, See https://github.com/yangheng95/PyABSA/issues/53
    SemEval = DatasetItem(
        "SemEval", ["113.Laptop14", "114.Restaurant14", "116.Restaurant16"]
    )
    Restaurant = DatasetItem("Restaurant", ["114.Restaurant14", "116.Restaurant16"])
    Multilingual = DatasetItem(
        "Multilingual",
        [
            "113.Laptop14",
            "114.Restaurant14",
            "116.Restaurant16",
            "101.ACL_Twitter",
            "109.MAMS",
            "117.Television",
            "118.TShirt",
            "119.Yelp",
            "107.Phone",
            "103.Camera",
            "106.Notebook",
            "104.Car",
            "105.MOOC",
            "129.Kaggle",
            "120.SemEval2016Task5",
            "121.MOOC_En",
            "130.Chinese_Zhang",
        ],
    )

    def __init__(self):
        super(APCDatasetList, self).__init__(
            [
                self.Laptop14,
                self.Restaurant14,
                self.ARTS_Laptop14,
                self.ARTS_Restaurant14,
                self.Restaurant15,
                self.Restaurant16,
                self.ACL_Twitter,
                self.MAMS,
                self.Television,
                self.TShirt,
                self.Yelp,
                self.Phone,
                self.Car,
                self.Notebook,
                self.Camera,
                self.Shampoo,
                self.MOOC,
                self.MOOC_En,
                self.Kaggle,
                self.Chinese_Zhang,
                self.Chinese,
                self.Binary_Polarity_Chinese,
                self.Triple_Polarity_Chinese,
                self.SemEval2016Task5,
                self.Arabic_SemEval2016Task5,
                self.Dutch_SemEval2016Task5,
                self.Spanish_SemEval2016Task5,
                self.Turkish_SemEval2016Task5,
                self.Russian_SemEval2016Task5,
                self.French_SemEval2016Task5,
                self.English_SemEval2016Task5,
                self.English,
                self.SemEval,
                self.Restaurant,
                self.Multilingual,
            ]
        )
