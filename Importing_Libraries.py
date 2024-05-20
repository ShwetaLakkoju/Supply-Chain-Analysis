{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "ae1eabdc-eff1-481b-b5dc-29bc097cd464",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: scikit-learn in /opt/conda/lib/python3.11/site-packages (1.4.2)\n",
      "Requirement already satisfied: numpy>=1.19.5 in /opt/conda/lib/python3.11/site-packages (from scikit-learn) (1.26.3)\n",
      "Requirement already satisfied: scipy>=1.6.0 in /opt/conda/lib/python3.11/site-packages (from scikit-learn) (1.13.0)\n",
      "Requirement already satisfied: joblib>=1.2.0 in /opt/conda/lib/python3.11/site-packages (from scikit-learn) (1.4.0)\n",
      "Requirement already satisfied: threadpoolctl>=2.0.0 in /opt/conda/lib/python3.11/site-packages (from scikit-learn) (3.4.0)\n"
     ]
    }
   ],
   "source": [
    "!pip install scikit-learn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "6c42e33f-3dc5-46d1-a55a-a1777903ff8b",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "import numpy as np \n",
    "import pandas as pd \n",
    "from sklearn.impute import KNNImputer\n",
    "from sklearn.preprocessing import OneHotEncoder\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from scipy.stats import skew, kurtosis\n",
    "from scipy.stats import zscore\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from scipy.stats import zscore\n",
    "from IPython.display import display\n",
    "from scipy.stats.mstats import winsorize\n",
    "import os"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "4f3ebbd9-072e-47fe-9e2f-a9d600aec6a0",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "df = pd.read_csv(\"./Supply_Chain_Shipment_Pricing_Dataset_20240325.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "f0733117-00fa-4cbd-8747-3e91a862d9ed",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Current Working Directory:  /home/jovyan/library/ist652/spring24/project\n"
     ]
    }
   ],
   "source": [
    "print(\"Current Working Directory: \", os.getcwd())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "16425b0b-c38c-4b81-ac9f-625f7469d5cb",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 10324 entries, 0 to 10323\n",
      "Data columns (total 33 columns):\n",
      " #   Column                        Non-Null Count  Dtype  \n",
      "---  ------                        --------------  -----  \n",
      " 0   id                            10324 non-null  int64  \n",
      " 1   project code                  10324 non-null  object \n",
      " 2   pq #                          10324 non-null  object \n",
      " 3   po / so #                     10324 non-null  object \n",
      " 4   asn/dn #                      10324 non-null  object \n",
      " 5   country                       10324 non-null  object \n",
      " 6   managed by                    10324 non-null  object \n",
      " 7   fulfill via                   10324 non-null  object \n",
      " 8   vendor inco term              10324 non-null  object \n",
      " 9   shipment mode                 9964 non-null   object \n",
      " 10  pq first sent to client date  10324 non-null  object \n",
      " 11  po sent to vendor date        10324 non-null  object \n",
      " 12  scheduled delivery date       10324 non-null  object \n",
      " 13  delivered to client date      10324 non-null  object \n",
      " 14  delivery recorded date        10324 non-null  object \n",
      " 15  product group                 10324 non-null  object \n",
      " 16  sub classification            10324 non-null  object \n",
      " 17  vendor                        10324 non-null  object \n",
      " 18  item description              10324 non-null  object \n",
      " 19  molecule/test type            10324 non-null  object \n",
      " 20  brand                         10324 non-null  object \n",
      " 21  dosage                        8588 non-null   object \n",
      " 22  dosage form                   10324 non-null  object \n",
      " 23  unit of measure (per pack)    10324 non-null  int64  \n",
      " 24  line item quantity            10324 non-null  int64  \n",
      " 25  line item value               10324 non-null  float64\n",
      " 26  pack price                    10324 non-null  float64\n",
      " 27  unit price                    10324 non-null  float64\n",
      " 28  manufacturing site            10324 non-null  object \n",
      " 29  first line designation        10324 non-null  bool   \n",
      " 30  weight (kilograms)            10324 non-null  object \n",
      " 31  freight cost (usd)            10324 non-null  object \n",
      " 32  line item insurance (usd)     10037 non-null  float64\n",
      "dtypes: bool(1), float64(4), int64(3), object(25)\n",
      "memory usage: 2.5+ MB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
