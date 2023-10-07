{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "58e28f9e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Looking in indexes: https://pypi.org/simple, https://pypi.ngc.nvidia.com\n",
      "Requirement already satisfied: datasets in /usr/local/lib/python3.8/dist-packages (2.14.4)\n",
      "Requirement already satisfied: packaging in /usr/local/lib/python3.8/dist-packages (from datasets) (22.0)\n",
      "Requirement already satisfied: aiohttp in /usr/local/lib/python3.8/dist-packages (from datasets) (3.8.5)\n",
      "Requirement already satisfied: dill<0.3.8,>=0.3.0 in /usr/local/lib/python3.8/dist-packages (from datasets) (0.3.7)\n",
      "Requirement already satisfied: multiprocess in /usr/local/lib/python3.8/dist-packages (from datasets) (0.70.15)\n",
      "Requirement already satisfied: huggingface-hub<1.0.0,>=0.14.0 in /usr/local/lib/python3.8/dist-packages (from datasets) (0.16.4)\n",
      "Requirement already satisfied: fsspec[http]>=2021.11.1 in /usr/local/lib/python3.8/dist-packages (from datasets) (2022.11.0)\n",
      "Requirement already satisfied: pyarrow>=8.0.0 in /usr/local/lib/python3.8/dist-packages (from datasets) (9.0.0)\n",
      "Requirement already satisfied: xxhash in /usr/local/lib/python3.8/dist-packages (from datasets) (3.3.0)\n",
      "Requirement already satisfied: tqdm>=4.62.1 in /usr/local/lib/python3.8/dist-packages (from datasets) (4.64.1)\n",
      "Requirement already satisfied: requests>=2.19.0 in /usr/local/lib/python3.8/dist-packages (from datasets) (2.28.2)\n",
      "Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.8/dist-packages (from datasets) (6.0)\n",
      "Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.8/dist-packages (from datasets) (1.22.2)\n",
      "Requirement already satisfied: pandas in /usr/local/lib/python3.8/dist-packages (from datasets) (1.5.2)\n",
      "Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.8/dist-packages (from aiohttp->datasets) (6.0.4)\n",
      "Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.8/dist-packages (from aiohttp->datasets) (1.4.0)\n",
      "Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.8/dist-packages (from aiohttp->datasets) (1.3.1)\n",
      "Requirement already satisfied: yarl<2.0,>=1.0 in /usr/local/lib/python3.8/dist-packages (from aiohttp->datasets) (1.9.2)\n",
      "Requirement already satisfied: async-timeout<5.0,>=4.0.0a3 in /usr/local/lib/python3.8/dist-packages (from aiohttp->datasets) (4.0.2)\n",
      "Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.8/dist-packages (from aiohttp->datasets) (22.2.0)\n",
      "Requirement already satisfied: charset-normalizer<4.0,>=2.0 in /usr/local/lib/python3.8/dist-packages (from aiohttp->datasets) (3.0.1)\n",
      "Requirement already satisfied: filelock in /usr/local/lib/python3.8/dist-packages (from huggingface-hub<1.0.0,>=0.14.0->datasets) (3.9.0)\n",
      "Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.8/dist-packages (from huggingface-hub<1.0.0,>=0.14.0->datasets) (4.5.0)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.8/dist-packages (from requests>=2.19.0->datasets) (2022.12.7)\n",
      "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.8/dist-packages (from requests>=2.19.0->datasets) (3.4)\n",
      "Requirement already satisfied: urllib3<1.27,>=1.21.1 in /usr/local/lib/python3.8/dist-packages (from requests>=2.19.0->datasets) (1.26.13)\n",
      "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.8/dist-packages (from pandas->datasets) (2022.6)\n",
      "Requirement already satisfied: python-dateutil>=2.8.1 in /usr/local/lib/python3.8/dist-packages (from pandas->datasets) (2.8.2)\n",
      "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.8/dist-packages (from python-dateutil>=2.8.1->pandas->datasets) (1.16.0)\n",
      "\u001b[33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv\u001b[0m\n",
      "\u001b[33mWARNING: You are using pip version 21.2.4; however, version 23.2.1 is available.\n",
      "You should consider upgrading via the '/usr/bin/python -m pip install --upgrade pip' command.\u001b[0m\n"
     ]
    }
   ],
   "source": [
    "!pip install datasets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "c3bc6aad",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "from datasets import load_dataset\n",
    "\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "dc3bf43f",
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset_dair_ai = load_dataset('dair-ai/emotion')\n",
    "\n",
    "train_dair_ai = dataset_dair_ai['train']\n",
    "test_dair_ai = dataset_dair_ai['test']\n",
    "validation_dair_ai = dataset_dair_ai['validation']\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "e84df721",
   "metadata": {},
   "outputs": [],
   "source": [
    "class_labels = train_dair_ai.features['label']\n",
    "\n",
    "dictionary_label = {0:'sadness',\n",
    " 1: 'joy',\n",
    " 2: 'love',\n",
    " 3: 'anger',\n",
    " 4: 'fear',\n",
    " 5: 'surprise'}\n",
    "\n",
    "\n",
    "train_dair_ai_pd = train_dair_ai.to_pandas()\n",
    "test_dair_ai_pd = test_dair_ai.to_pandas()\n",
    "validation_dair_ai_pd = validation_dair_ai.to_pandas()\n",
    "\n",
    "\n",
    "train_dair_ai_pd['label'] = train_dair_ai_pd['label'].apply(lambda x: dictionary_label[x])\n",
    "test_dair_ai_pd['label'] = test_dair_ai_pd['label'].apply(lambda x: dictionary_label[x])\n",
    "validation_dair_ai_pd['label'] = validation_dair_ai_pd['label'].apply(lambda x: dictionary_label[x])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "befef913",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "joy         5362\n",
       "sadness     4666\n",
       "anger       2159\n",
       "fear        1937\n",
       "love        1304\n",
       "surprise     572\n",
       "Name: label, dtype: int64"
      ]
     },
     "execution_count": 57,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_dair_ai_pd['label'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "b61e9cae",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "joy         695\n",
       "sadness     581\n",
       "anger       275\n",
       "fear        224\n",
       "love        159\n",
       "surprise     66\n",
       "Name: label, dtype: int64"
      ]
     },
     "execution_count": 58,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_dair_ai_pd['label'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "b38f2fe8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "joy         704\n",
       "sadness     550\n",
       "anger       275\n",
       "fear        212\n",
       "love        178\n",
       "surprise     81\n",
       "Name: label, dtype: int64"
      ]
     },
     "execution_count": 59,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "validation_dair_ai_pd['label'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "f9b5388d",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_dair_ai_pd.rename(columns={\"text\": \"Text_processed\", \"label\": \"Emotion\"}, inplace=True)\n",
    "test_dair_ai_pd.rename(columns={\"text\": \"Text_processed\", \"label\": \"Emotion\"}, inplace=True)\n",
    "validation_dair_ai_pd.rename(columns={\"text\": \"Text_processed\", \"label\": \"Emotion\"}, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "cf1b9b05",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Drop rows where label is equal to 'love' or 'surprise'\n",
    "filtered_train_dair_ai_pd = train_dair_ai_pd[(train_dair_ai_pd['Emotion'] != 'love') & (train_dair_ai_pd['Emotion'] != 'surprise')]\n",
    "filtered_test_dair_ai_pd = test_dair_ai_pd[(test_dair_ai_pd['Emotion'] != 'love') & (test_dair_ai_pd['Emotion'] != 'surprise')]\n",
    "filtered_validation_dair_ai_pd = validation_dair_ai_pd[(validation_dair_ai_pd['Emotion'] != 'love') & (validation_dair_ai_pd['Emotion'] != 'surprise')]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "d36514ba",
   "metadata": {},
   "outputs": [],
   "source": [
    "all_dair_ai = pd.concat([filtered_train_dair_ai_pd, filtered_test_dair_ai_pd, filtered_validation_dair_ai_pd])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "3b856586",
   "metadata": {},
   "outputs": [],
   "source": [
    "all_dair_ai['text'] = all_dair_ai.apply(lambda x: f\"### Human: Now I want you to perform a classification of the following sentence based on the emotion it represents, you can use Anger, Joy, Sadness, Guilt, Shame, Fear, and Disgust. {x['Text_processed']} ### Assistant: {x['Emotion']}\", axis=1)\n",
    "all_dair_ai['Augmented'] = False"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "a367543e",
   "metadata": {},
   "outputs": [],
   "source": [
    "from datasets import Dataset\n",
    "\n",
    "dataset_dair_ai = Dataset.from_pandas(all_dair_ai)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "2372614a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Dataset({\n",
       "    features: ['Text_processed', 'Emotion', 'text', 'Augmented'],\n",
       "    num_rows: 17640\n",
       "})"
      ]
     },
     "execution_count": 65,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset_dair_ai.remove_columns('__index_level_0__')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "540fb0c8",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Pushing dataset shards to the dataset hub:   0%|                                                                                                                                                                                                                                                        | 0/1 [00:00<?, ?it/s]\n",
      "Creating parquet from Arrow format: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 18/18 [00:00<00:00, 555.35ba/s]\u001b[A\n",
      "Pushing dataset shards to the dataset hub: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.88s/it]\n"
     ]
    }
   ],
   "source": [
    "dataset_dair_ai.push_to_hub('RikoteMaster/dataset_dair_ai_4_llama2_v3')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "58d052f1",
   "metadata": {},
   "outputs": [],
   "source": [
    "goemotion_dataset = pd.read_csv('goemotions_selected_emotions.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "c5254496",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "anger            8084\n",
       "joy              7983\n",
       "sadness          6758\n",
       "disgust          5301\n",
       "fear             3197\n",
       "remorse          2525\n",
       "embarrassment    2476\n",
       "Name: label, dtype: int64"
      ]
     },
     "execution_count": 68,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "goemotion_dataset['label'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "896f6329",
   "metadata": {},
   "outputs": [],
   "source": [
    "goemotion_dataset['label'] = goemotion_dataset['label'].replace({'remorse': 'guilt', 'embarrassment': 'shame'})\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "id": "f5709f44",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "anger      8084\n",
       "joy        7983\n",
       "sadness    6758\n",
       "disgust    5301\n",
       "fear       3197\n",
       "guilt      2525\n",
       "shame      2476\n",
       "Name: label, dtype: int64"
      ]
     },
     "execution_count": 70,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "goemotion_dataset['label'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "ce2c1b20",
   "metadata": {},
   "outputs": [],
   "source": [
    "goemotion_dataset.rename(columns={\"text\": \"Text_processed\", \"label\": \"Emotion\"}, inplace=True)\n",
    "goemotion_dataset['text'] = goemotion_dataset.apply(lambda x: f\"### Human: Now I want you to perform a classification of the following sentence based on the emotion it represents, you can use Anger, Joy, Sadness, Guilt, Shame, Fear, and Disgust. {x['Text_processed']} ### Assistant: {x['Emotion']}\", axis=1)\n",
    "goemotion_dataset['Augmented'] = False"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "id": "f03d1708",
   "metadata": {},
   "outputs": [],
   "source": [
    "goemotion_dataset = goemotion_dataset.sample(frac=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "id": "9610e9de",
   "metadata": {},
   "outputs": [],
   "source": [
    "hf_goemotion_dataset = Dataset.from_pandas(goemotion_dataset)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "id": "980044a7",
   "metadata": {},
   "outputs": [],
   "source": [
    "hf_goemotion_dataset = hf_goemotion_dataset.remove_columns('__index_level_0__')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "id": "00b2396d",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Pushing dataset shards to the dataset hub:   0%|                                                                                                                                                                                                                                                        | 0/1 [00:00<?, ?it/s]\n",
      "Creating parquet from Arrow format: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 37/37 [00:00<00:00, 724.77ba/s]\u001b[A\n",
      "Pushing dataset shards to the dataset hub: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.68s/it]\n"
     ]
    }
   ],
   "source": [
    "hf_goemotion_dataset.push_to_hub('goemotion_4_llama2_v3')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "id": "014cd18d",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Downloading readme: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 616/616 [00:00<00:00, 990kB/s]\n"
     ]
    }
   ],
   "source": [
    "from datasets import load_dataset\n",
    "\n",
    "isear_dataset = load_dataset('RikoteMaster/isear_for_llama2')\n",
    "dair_ai_dataset = load_dataset('RikoteMaster/dataset_dair_ai_4_llama2_v2')\n",
    "goemotion_dataset = load_dataset('RikoteMaster/goemotion_4_llama2_v2')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "id": "6bf3bc9e",
   "metadata": {},
   "outputs": [],
   "source": [
    "dair_ai_dataset = dair_ai_dataset.remove_columns('__index_level_0__')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "id": "036ec8e2",
   "metadata": {},
   "outputs": [],
   "source": [
    "from datasets import concatenate_datasets\n",
    "\n",
    "macro_ds = concatenate_datasets([isear_dataset['train'], dair_ai_dataset['train'], goemotion_dataset['train']])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "id": "48930ec3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Dataset({\n",
       "    features: ['Text_processed', 'Emotion', 'Augmented', 'text'],\n",
       "    num_rows: 61463\n",
       "})"
      ]
     },
     "execution_count": 92,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "macro_ds"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "id": "59137597",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Pushing dataset shards to the dataset hub:   0%|                                                                                                                                                                                                                                                        | 0/1 [00:00<?, ?it/s]\n",
      "Creating parquet from Arrow format: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 62/62 [00:00<00:00, 623.97ba/s]\u001b[A\n",
      "Pushing dataset shards to the dataset hub: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:02<00:00,  2.50s/it]\n"
     ]
    }
   ],
   "source": [
    "macro_ds.push_to_hub('Emotion_Recognition_4_llama2')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 96,
   "id": "901cccf0",
   "metadata": {},
   "outputs": [],
   "source": [
    "macro_ds_pd = macro_ds.to_pandas()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 103,
   "id": "2b121549",
   "metadata": {},
   "outputs": [],
   "source": [
    "macro_emotions = macro_ds_pd['Emotion']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 108,
   "id": "0b0f506d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "joy        15859\n",
       "sadness    13627\n",
       "anger      11872\n",
       "fear        6653\n",
       "disgust     6406\n",
       "guilt       3543\n",
       "shame       3503\n",
       "Name: Emotion, dtype: int64"
      ]
     },
     "execution_count": 108,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "macro_emotions.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 109,
   "id": "947ce2a0",
   "metadata": {},
   "outputs": [
    {
     "ename": "IndentationError",
     "evalue": "unexpected indent (2899810926.py, line 2)",
     "output_type": "error",
     "traceback": [
      "\u001b[0;36m  Cell \u001b[0;32mIn[109], line 2\u001b[0;36m\u001b[0m\n\u001b[0;31m    --load_in_4bit \\\u001b[0m\n\u001b[0m    ^\u001b[0m\n\u001b[0;31mIndentationError\u001b[0m\u001b[0;31m:\u001b[0m unexpected indent\n"
     ]
    }
   ],
   "source": [
    "!python trl/examples/scripts/sft_trainer.py \\\n",
    "    --model_name meta-llama/Llama-2-7b-hf \\\n",
    "    --dataset_name RikoteMaster/isear_for_llama2 \\\n",
    "    --output_dir ./model\n",
    "    --load_in_4bit \\\n",
    "    --use_peft \\\n",
    "    --batch_size 8 \\\n",
    "    --gradient_accumulation_steps 2\n",
    "    \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "995434c2",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.8/dist-packages/tqdm/auto.py:22: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html\n",
      "  from .autonotebook import tqdm as notebook_tqdm\n"
     ]
    }
   ],
   "source": [
    "from datasets import load_dataset\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "8bc5015b",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Map: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61463/61463 [00:02<00:00, 29729.98 examples/s]\n"
     ]
    }
   ],
   "source": [
    "\n",
    "ds = load_dataset('RikoteMaster/Emotion_Recognition_4_llama2_v2')\n",
    "\n",
    "def bigger_formatting(ds):\n",
    "    ds['text'] = f\"\"\"###Human:\\nIn this task, you will be performing a classification exercise aimed at identifying the underlying emotion conveyed by a given sentence. The emotions to consider are as follows:\n",
    "\n",
    "    Joy: Joy is a positive and uplifting emotion characterized by happiness, elation, and a sense of contentment. It arises from pleasant experiences, achievements, or connections with others.\n",
    "\n",
    "    Sadness: Sadness is a feeling of sorrow, unhappiness, or despondency. It is often triggered by loss, disappointment, or a sense of longing.\n",
    "\n",
    "    Guilt: Guilt is a self-directed emotion that arises from a sense of wrongdoing or moral transgression. It involves feeling responsible for a negative outcome or harm done to others.\n",
    "\n",
    "    Shame: Shame is a powerful emotion associated with feeling embarrassed, humiliated, or unworthy. It typically arises from a perception of public exposure of one's flaws or mistakes.\n",
    "\n",
    "    Fear: Fear is an emotion triggered by a perceived threat or danger. It can lead to a heightened state of alertness, anxiety, and a desire to avoid the source of fear.\n",
    "\n",
    "    Disgust: Disgust is an aversive emotion linked to feelings of revulsion, repulsion, or strong distaste. It arises in response to things that are offensive or unpleasant.\n",
    "\n",
    "    Anger: Anger is a strong feeling of displeasure, hostility, or frustration. It often arises when one's boundaries, values, or rights are violated, leading to a desire for confrontation or retaliation.\n",
    "    \n",
    "    Your task is to analyze each sentence provided and categorize it into one of these emotions based on the dominant feeling conveyed by the text. This classification will require an understanding of the nuances of human emotions and the context in which the sentences are presented.\n",
    "        \n",
    "    Remember, you have to classify the sentences using only Anger, Joy, Sadnes, Guilt, Shame, fear or disgust\n",
    "    \n",
    "    Sentence: {ds['Text_processed']}\\n\\n###Assistant:\\n{ds['Emotion']}\"\"\"\n",
    "    \n",
    "\n",
    "ds = ds.map(bigger_formatting)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "b8fccdcb",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Pushing dataset shards to the dataset hub:   0%|                                                                                                                                                                                                                                         | 0/1 [00:00<?, ?it/s]\n",
      "Creating parquet from Arrow format:   0%|                                                                                                                                                                                                                                               | 0/62 [00:00<?, ?ba/s]\u001b[A\n",
      "Creating parquet from Arrow format: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 62/62 [00:00<00:00, 408.01ba/s]\u001b[A\n",
      "Pushing dataset shards to the dataset hub: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:02<00:00,  2.66s/it]\n"
     ]
    }
   ],
   "source": [
    "ds.push_to_hub('RikoteMaster/Emotion_Recognition_4_llama2_v2')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "f42d1f93",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Map: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7499/7499 [00:00<00:00, 15362.43 examples/s]\n",
      "Map: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1324/1324 [00:00<00:00, 17803.70 examples/s]\n",
      "Map: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1879/1879 [00:00<00:00, 17932.21 examples/s]\n"
     ]
    }
   ],
   "source": [
    "ds = load_dataset(\"RikoteMaster/isear_for_llama2\")\n",
    "def bigger_formatting(ds):\n",
    "    ds['text'] = f\"\"\"###Human:\\nIn this task, you will be performing a classification exercise aimed at identifying the underlying emotion conveyed by a given sentence. The emotions to consider are as follows:\n",
    "\n",
    "    Joy: Joy is a positive and uplifting emotion characterized by happiness, elation, and a sense of contentment. It arises from pleasant experiences, achievements, or connections with others.\n",
    "\n",
    "    Sadness: Sadness is a feeling of sorrow, unhappiness, or despondency. It is often triggered by loss, disappointment, or a sense of longing.\n",
    "\n",
    "    Guilt: Guilt is a self-directed emotion that arises from a sense of wrongdoing or moral transgression. It involves feeling responsible for a negative outcome or harm done to others.\n",
    "\n",
    "    Shame: Shame is a powerful emotion associated with feeling embarrassed, humiliated, or unworthy. It typically arises from a perception of public exposure of one's flaws or mistakes.\n",
    "\n",
    "    Fear: Fear is an emotion triggered by a perceived threat or danger. It can lead to a heightened state of alertness, anxiety, and a desire to avoid the source of fear.\n",
    "\n",
    "    Disgust: Disgust is an aversive emotion linked to feelings of revulsion, repulsion, or strong distaste. It arises in response to things that are offensive or unpleasant.\n",
    "\n",
    "    Anger: Anger is a strong feeling of displeasure, hostility, or frustration. It often arises when one's boundaries, values, or rights are violated, leading to a desire for confrontation or retaliation.\n",
    "    \n",
    "    Your task is to analyze each sentence provided and categorize it into one of these emotions based on the dominant feeling conveyed by the text. This classification will require an understanding of the nuances of human emotions and the context in which the sentences are presented.\n",
    "        \n",
    "    Remember, you have to classify the sentences using only Anger, Joy, Sadnes, Guilt, Shame, fear or disgust\n",
    "    \n",
    "    Sentence: {ds['Text_processed']}\\n\\n###Assistant:\\n{ds['Emotion']}\"\"\"\n",
    "    \n",
    "    return ds\n",
    "\n",
    "ds = ds.map(bigger_formatting)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "1d1fc0af",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Pushing dataset shards to the dataset hub:   0%|                                                                                                                                                                                                                                                 | 0/1 [00:00<?, ?it/s]\n",
      "Creating parquet from Arrow format: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 315.48ba/s]\u001b[A\n",
      "Pushing dataset shards to the dataset hub: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.99s/it]\n",
      "Pushing dataset shards to the dataset hub:   0%|                                                                                                                                                                                                                                                 | 0/1 [00:00<?, ?it/s]\n",
      "Creating parquet from Arrow format: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 401.95ba/s]\u001b[A\n",
      "Pushing dataset shards to the dataset hub: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.04s/it]\n",
      "Pushing dataset shards to the dataset hub:   0%|                                                                                                                                                                                                                                                 | 0/1 [00:00<?, ?it/s]\n",
      "Creating parquet from Arrow format: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 321.45ba/s]\u001b[A\n",
      "Pushing dataset shards to the dataset hub: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.22s/it]\n"
     ]
    }
   ],
   "source": [
    "ds.push_to_hub(\"RikoteMaster/isear_for_llama2_v3\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "c39b47dc",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Map: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61463/61463 [00:03<00:00, 19853.20 examples/s]\n",
      "Pushing dataset shards to the dataset hub:   0%|                                                                                                                                                                                                                                                 | 0/1 [00:00<?, ?it/s]\n",
      "Creating parquet from Arrow format: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 62/62 [00:00<00:00, 620.48ba/s]\u001b[A\n",
      "Pushing dataset shards to the dataset hub: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:02<00:00,  2.34s/it]\n"
     ]
    }
   ],
   "source": [
    "from datasets import load_dataset\n",
    "\n",
    "ds = load_dataset(\"RikoteMaster/Emotion_Recognition_4_llama2_v2\")\n",
    "def bigger_formatting(ds):\n",
    "    ds['text'] = f\"\"\"###Human:\\nIn this task, you will be performing a classification exercise aimed at identifying the underlying emotion conveyed by a given sentence. The emotions to consider are as follows:\n",
    "\n",
    "    Anger, Joy, Sadnes, Guilt, Shame, fear or disgust\n",
    "    \n",
    "    Sentence: {ds['Text_processed']}\\n\\n###Assistant:\\n{ds['Emotion']}\"\"\"\n",
    "    \n",
    "    return ds\n",
    "\n",
    "ds = ds.map(bigger_formatting)\n",
    "\n",
    "ds.push_to_hub(\"RikoteMaster/Emotion_Recognition_4_llama2_v3\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "70d436e2",
   "metadata": {},
   "source": [
    "\"<s>[INST] Me gradué hace poco de la carrera de medicina ¿Me podrías aconsejar para conseguir rápidamente un puesto de trabajo? [/INST] Esto vale tanto para médicos como para cualquier otra profesión tras finalizar los estudios aniversarios y mi consejo sería preguntar a cuántas personas haya conocido mejor. En este caso, mi primera opción sería hablar con otros profesionales médicos, echar currículos en hospitales y cualquier centro de salud. En paralelo, trabajaría por mejorar mi marca personal como médico mediante un blog o formas digitales de comunicación como los vídeos. Y, para mejorar las posibilidades de encontrar trabajo, también participaría en congresos y encuentros para conseguir más contactos. Y, además de todo lo anterior, seguiría estudiando para presentarme a las oposiciones y ejercer la medicina en el sector público de mi país. </s>\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "5e14cc27",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.8/dist-packages/tqdm/auto.py:22: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html\n",
      "  from .autonotebook import tqdm as notebook_tqdm\n",
      "Map: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61463/61463 [00:03<00:00, 19205.84 examples/s]\n",
      "Pushing dataset shards to the dataset hub:   0%|                                                                                                                                                                                                                                                 | 0/1 [00:00<?, ?it/s]\n",
      "Creating parquet from Arrow format:   0%|                                                                                                                                                                                                                                                       | 0/62 [00:00<?, ?ba/s]\u001b[A\n",
      "Creating parquet from Arrow format: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 62/62 [00:00<00:00, 607.11ba/s]\u001b[A\n",
      "Pushing dataset shards to the dataset hub: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:02<00:00,  2.53s/it]\n"
     ]
    }
   ],
   "source": [
    "from datasets import load_dataset\n",
    "\n",
    "ds = load_dataset(\"RikoteMaster/Emotion_Recognition_4_llama2_v2\")\n",
    "def bigger_formatting(ds):\n",
    "    ds['text'] = f\"\"\"<s>[INST] In this task, you will be performing a classification exercise aimed at identifying the underlying emotion conveyed by a given sentence. The emotions to consider are as follows:\n",
    "\n",
    "    Anger, Joy, Sadnes, Guilt, Shame, fear or disgust\n",
    "    \n",
    "    Sentence: {ds['Text_processed']} [/INST] {ds['Emotion']} <s>\"\"\"\n",
    "    \n",
    "    return ds\n",
    "\n",
<<<<<<< HEAD
    "ds = ds.map(bigger_formatting)a\n",
=======
    "ds = ds.map(bigger_formatting)\n",
>>>>>>> d838da85df027458b7fc9c45cfb23edae147628c
    "\n",
    "ds.push_to_hub(\"RikoteMaster/Emotion_Recognition_4_llama2_chat\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a5e15f56",
   "metadata": {},
   "outputs": [],
   "source": []
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
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}