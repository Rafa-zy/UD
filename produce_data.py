
from ast import Break
import os
import torch
from datasets import concatenate_datasets, load_dataset,Dataset, load_from_disk
import numpy as np

#import torch.utils.data as data_utils
#from data_utils import T0_TRAIN_TASK_LIST,T0_TEST_TASK_LIST

from transformers import AutoModel, AutoTokenizer
from transformers import T5Tokenizer
import csv
import json
import re
import random

import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]="2"


T0_TRAIN_TASK_LIST = [
    "glue/mrpc",
    "glue/qqp",
    "paws/labeled_final",
    "ag_news",
    "dbpedia_14",
    "dream",
    "kilt_tasks/hotpotqa",
    "trec",
    "cnn_dailymail/3.0.0",
    "gigaword",
    "multi_news",
    "samsum",
    "xsum",
    "amazon_polarity",
    "app_reviews",
    "imdb",
    "rotten_tomatoes",
    "yelp_review_full",
    "wiki_qa",
    "common_gen",
    "wiki_bio",
    "adversarial_qa/dbidaf",
    "adversarial_qa/dbert",
    "adversarial_qa/droberta",
    "quoref",
    "ropes",
    "duorc/SelfRC",
    "duorc/ParaphraseRC",
    "wiki_hop/original",
    "sciq",
    "quarel",
    "qasc",
    "cosmos_qa",
    "wiqa",
    "social_i_qa",
    "quail",
    "quartz",
    "cos_e/v1.11",
]

all_task_family = [
    'coreference_resolution', 'natural_language_inference', 
    'paraphrase_identification', 'closed_book_qa', 'extractive_qa', 
    'multiple_choice_qa', 'sentiment', 'sentence_completion', 
    'structure_to_text', 'summarization', 'topic_classification', 'word_sense_disambiguation']
train_task_family = [
    'paraphrase_identification', 'closed_book_qa', 'extractive_qa', 
    'multiple_choice_qa', 'sentiment', 'structure_to_text', 'summarization', 
    'topic_classification']
test_task_family = [
    'coreference_resolution', 'natural_language_inference', 
    'sentence_completion', 'word_sense_disambiguation']
TASK_TYPE_DICT = {
    "coreference_resolution": [
        "super_glue/wsc.fixed", "winogrande/winogrande_xl"
    ],
    "natural_language_inference":[
        "super_glue/cb", "super_glue/rte", "anli"
    ],
    "paraphrase_identification":[
        "glue/mrpc", "glue/qqp", "paws/labeled_final"
    ],
    "closed_book_qa":[
        # "ai2_arc/ARC Challenge",
        # "ai2_arc/ARC_Easy",
        "kilt_tasks/hotpotqa",
        # "trivia_qa/unfiltered",
        # "web_questions",
        "wiki_qa"
    ],
    "extractive_qa":[
        "adversarial_qa/dbidaf",
        "adversarial_qa/dbert",
        "adversarial_qa/droberta",
        "duorc/SelfRC",
        "duorc/ParaphraseRC",
        "ropes",
        "quoref" #xhk: new added
        ],
    "multiple_choice_qa":[
        #"commonsense_qa", 
        "cos_e/v1.11", # xhk: replace the previous one
        "cosmos_qa",
        "dream",
        "qasc",
        "quail",
        "quarel",
        "quartz",
        "sciq",
        "social_i_qa",
        "wiki_hop/original",
        "wiqa",
        ],
    "sentiment": [
        "amazon_polarity", "app_reviews", "imdb", "rotten_tomatoes", "yelp_review_full"
    ],
    "sentence_completion": [
        "super_glue/copa", "story_cloze/2016", "hellaswag"
    ],
    "structure_to_text": [
        "common_gen", "wiki_bio"
    ],
    "summarization": [
        "cnn_dailymail/3.0.0", "gigaword", "multi_news", "samsum", "xsum"
    ],
    "topic_classification": [
        "ag_news", "dbpedia_14", "trec"
    ],
    "word_sense_disambiguation": [
        "super_glue/wic"
    ]
}

cls_template_names = ['ag_news_classify_question_first', 'ag_news_classify_with_choices_question_first',
                      'ag_news_recommend', 'ag_news_which_section_choices', 'ag_news_which_section',
                      'ag_news_classify_with_choices', 'ag_news_classify', 'app_reviews_categorize_rating_using_review',
                      'app_reviews_convert_to_star_rating', 'wiki_hop_original_choose_best_object_interrogative_1',
                      'wiki_hop_original_choose_best_object_affirmative_1',
                      'wiki_hop_original_choose_best_object_affirmative_3',
                      'wiki_hop_original_choose_best_object_affirmative_2',
                      'wiki_hop_original_choose_best_object_interrogative_2',
                      'glue_mrpc_want_to_know', 'glue_mrpc_paraphrase', 'glue_mrpc_equivalent',
                      'glue_mrpc_replace', 'glue_mrpc_same_thing', 'glue_qqp_quora', 'glue_qqp_duplicate_or_not',
                      'glue_qqp_same_thing', 'glue_qqp_answer', 'glue_qqp_meaning', 'glue_qqp_duplicate',
                      'amazon_polarity_Is_this_review', 'amazon_polarity_User_recommend_this_product',
                      'amazon_polarity_Is_this_product_review_positive', 'amazon_polarity_Is_this_review_negative',
                      'amazon_polarity_convey_negative_or_positive_sentiment', 'amazon_polarity_negative_or_positive_tone',
                      'amazon_polarity_user_satisfied', 'amazon_polarity_would_you_buy', 'amazon_polarity_flattering_or_not',
                      'paws_labeled_final_task_description_no_label', 'paws_labeled_final_Meaning',
                      'paws_labeled_final_context_question_no_label', 'paws_labeled_final_Rewrite_no_label',
                      'paws_labeled_final_context_question', 'paws_labeled_final_Concatenation',
                      'paws_labeled_final_Concatenation_no_label', 'paws_labeled_final_Meaning_no_label',
                      'paws_labeled_final_PAWS_ANLI_GPT3', 'paws_labeled_final_Rewrite',
                      'paws_labeled_final_PAWS_ANLI_GPT3_no_label',
                      'dbpedia_14_given_list_what_category_does_the_paragraph_belong_to',
                      'dbpedia_14_pick_one_category_for_the_following_text', 'dbpedia_14_given_a_choice_of_categories_',
                      'dbpedia_14_given_a_list_of_category_what_does_the_title_belong_to', 'dream_baseline',
                      'dream_read_the_following_conversation_and_answer_the_question', 'trec_what_category_best_describe',
                      'trec_fine_grained_ENTY', 'trec_pick_the_best_descriptor', 'trec_fine_grained_open_context_first',
                      'trec_which_category_best_describes', 'trec_trec1', 'trec_trec2', 'trec_fine_grained_open',
                      'imdb_Movie_Expressed_Sentiment_2', 'imdb_Reviewer_Opinion_bad_good_choices', 'imdb_Sentiment_with_choices_',
                      'imdb_Reviewer_Sentiment_Feeling', 'imdb_Writer_Expressed_Sentiment', 'imdb_Movie_Expressed_Sentiment',
                      'imdb_Text_Expressed_Sentiment', 'imdb_Reviewer_Enjoyment_Yes_No', 'imdb_Reviewer_Expressed_Sentiment',
                      'imdb_Reviewer_Enjoyment', 'rotten_tomatoes_Reviewer_Opinion_bad_good_choices', 'rotten_tomatoes_Text_Expressed_Sentiment', 'rotten_tomatoes_Sentiment_with_choices_',
                      'rotten_tomatoes_Reviewer_Enjoyment_Yes_No', 'rotten_tomatoes_Reviewer_Enjoyment', 'rotten_tomatoes_Movie_Expressed_Sentiment',
                      'rotten_tomatoes_Writer_Expressed_Sentiment', 'rotten_tomatoes_Movie_Expressed_Sentiment_2', 'rotten_tomatoes_Reviewer_Expressed_Sentiment',
                      'rotten_tomatoes_Reviewer_Sentiment_Feeling', 'yelp_review_full_so_i_would', 'yelp_review_full_based_on_that', 'yelp_review_full_format_star',
                      'yelp_review_full_this_place', 'yelp_review_full_format_score', 'yelp_review_full_on_a_scale', 'yelp_review_full_format_rating',
                      'wiki_qa_Is_This_True_', 'wiki_qa_automatic_system', 'wiki_qa_found_on_google', 'wiki_qa_exercise', 'wiki_qa_Decide_good_answer',
                      'sciq_Direct_Question_Closed_Book_', 'sciq_Multiple_Choice_Closed_Book_', 'sciq_Multiple_Choice_Question_First', 'sciq_Multiple_Choice',
                      'sciq_Direct_Question', 'quarel_do_not_use', 'quarel_logic_test', 'quarel_heres_a_story', 'quarel_choose_between',
                      'quarel_testing_students', 'qasc_is_correct_1', 'qasc_qa_with_separated_facts_1', 'qasc_qa_with_separated_facts_3',
                      'qasc_qa_with_separated_facts_4', 'qasc_qa_with_separated_facts_5', 'qasc_qa_with_combined_facts_1', 'qasc_is_correct_2',
                      'qasc_qa_with_separated_facts_2', 'cosmos_qa_description_context_question_answer_text', 'cosmos_qa_description_context_question_text',
                      'cosmos_qa_description_context_question_answer_id', 'cosmos_qa_context_description_question_answer_text', 'cosmos_qa_no_prompt_id',
                      'cosmos_qa_context_question_description_text', 'cosmos_qa_no_prompt_text', 'cosmos_qa_context_description_question_answer_id',
                      'cosmos_qa_context_question_description_answer_id', 'cosmos_qa_context_description_question_text',
                      'cosmos_qa_context_question_description_answer_text', 'cosmos_qa_only_question_answer', 
                      'social_i_qa_I_was_wondering', 'social_i_qa_Show_choices_and_generate_answer', 'social_i_qa_Check_if_a_random_answer_is_valid_or_not', 'social_i_qa_Generate_answer',
                      'social_i_qa_Show_choices_and_generate_index', 'quail_context_question_answer_description_id', 'quail_context_question_answer_description_text',
                      'quail_description_context_question_answer_id', 'quail_context_question_description_answer_text', 'quail_context_question_description_text',
                      'quail_context_description_question_text', 'quail_context_question_description_answer_id', 'quail_no_prompt_id', 'quail_context_description_question_answer_id',
                      'quail_description_context_question_text', 'quail_no_prompt_text', 'quail_context_description_question_answer_text', 'quail_description_context_question_answer_text',
                      'quartz_use_info_from_question_paragraph', 'quartz_paragraph_question_plain_concat', 'quartz_use_info_from_paragraph_question',
                      'quartz_answer_question_based_on', 'quartz_answer_question_below',
                      'quartz_read_passage_below_choose', 'quartz_having_read_above_passage', 'quartz_given_the_fact_answer_the_q', 'cos_e_v1.11_question_description_option_text',
                      'cos_e_v1.11_question_description_option_id', 'cos_e_v1.11_question_option_description_text',
                      'cos_e_v1.11_description_question_option_id', 'cos_e_v1.11_description_question_option_text', 'cos_e_v1.11_question_option_description_id',
'trec_fine_grained_LOC', 'trec_fine_grained_NUM_context_first', 'trec_fine_grained_NUM', 'trec_fine_grained_LOC_context_first',
                      'trec_fine_grained_DESC', 'trec_fine_grained_ABBR', 'trec_fine_grained_ABBR_context_first',
                      'trec_fine_grained_HUM', 'trec_fine_grained_HUM_context_first', 'trec_fine_grained_DESC_context_first'
                      ]

full_template_names=[   'adversarial_qa_dbert_answer_the_following_q', 'adversarial_qa_dbert_based_on', 'adversarial_qa_dbert_generate_question', 'adversarial_qa_dbert_question_context_answer', 
                        'adversarial_qa_dbert_tell_what_it_is', 'adversarial_qa_dbidaf_answer_the_following_q', 'adversarial_qa_dbidaf_based_on', 'adversarial_qa_dbidaf_generate_question', 
                        'adversarial_qa_dbidaf_question_context_answer', 'adversarial_qa_dbidaf_tell_what_it_is', 'adversarial_qa_droberta_answer_the_following_q', 'adversarial_qa_droberta_based_on', 
                        'adversarial_qa_droberta_generate_question', 'adversarial_qa_droberta_question_context_answer', 'adversarial_qa_droberta_tell_what_it_is', 'ag_news_classify', 
                        'ag_news_classify_question_first', 'ag_news_classify_with_choices', 'ag_news_classify_with_choices_question_first', 'ag_news_recommend', 'ag_news_which_section', 
                        'ag_news_which_section_choices', 'amazon_polarity_Is_this_product_review_positive', 'amazon_polarity_Is_this_review', 'amazon_polarity_Is_this_review_negative', 
                        'amazon_polarity_User_recommend_this_product', 'amazon_polarity_convey_negative_or_positive_sentiment', 'amazon_polarity_flattering_or_not', 
                        'amazon_polarity_negative_or_positive_tone', 'amazon_polarity_user_satisfied', 'amazon_polarity_would_you_buy', 'app_reviews_categorize_rating_using_review', 
                        'app_reviews_convert_to_rating', 'app_reviews_convert_to_star_rating', 'app_reviews_generate_review', 'cnn_dailymail_3.0.0_2_or_3_sentences', 
                        'cnn_dailymail_3.0.0_generate_story', 'cnn_dailymail_3.0.0_news_card_view', 'cnn_dailymail_3.0.0_news_stock', 'cnn_dailymail_3.0.0_news_summary', 
                        'cnn_dailymail_3.0.0_spice_up_story', 'cnn_dailymail_3.0.0_sum_in_brief', 'cnn_dailymail_3.0.0_tldr_summary', 'cnn_dailymail_3.0.0_write_an_outline', 
                        'common_gen_Example_prompt', 'common_gen_Given_concepts_type_1', 'common_gen_Given_concepts_type_2', 'common_gen_Put_together', 
                        'common_gen_choice_in_concept_centric_sentence_generation', 'common_gen_random_task_template_prompt', 'common_gen_sentence_to_concepts', 'common_gen_topic_to_sentence', 
                        'common_gen_topics_from_the_sentence', 'cos_e_v1.11_aligned_with_common_sense', 'cos_e_v1.11_description_question_option_id', 'cos_e_v1.11_description_question_option_text', 
                        'cos_e_v1.11_explain_why_human', 'cos_e_v1.11_generate_explanation_given_text', 'cos_e_v1.11_i_think', 'cos_e_v1.11_question_description_option_id', 
                        'cos_e_v1.11_question_description_option_text', 'cos_e_v1.11_question_option_description_id', 'cos_e_v1.11_question_option_description_text', 'cos_e_v1.11_rationale', 
                        'cosmos_qa_context_answer_to_question', 'cosmos_qa_context_description_question_answer_id', 'cosmos_qa_context_description_question_answer_text', 
                        'cosmos_qa_context_description_question_text', 'cosmos_qa_context_question_description_answer_id', 'cosmos_qa_context_question_description_answer_text', 
                        'cosmos_qa_context_question_description_text', 'cosmos_qa_description_context_question_answer_id', 'cosmos_qa_description_context_question_answer_text', 
                        'cosmos_qa_description_context_question_text', 'cosmos_qa_no_prompt_id', 'cosmos_qa_no_prompt_text', 'cosmos_qa_only_question_answer', 
                        'dbpedia_14_given_a_choice_of_categories_', 'dbpedia_14_given_a_list_of_category_what_does_the_title_belong_to', 
                        'dbpedia_14_given_list_what_category_does_the_paragraph_belong_to', 'dbpedia_14_pick_one_category_for_the_following_text', 'dream_answer_to_dialogue', 'dream_baseline', 
                        'dream_generate_first_utterance', 'dream_generate_last_utterance', 'dream_read_the_following_conversation_and_answer_the_question', 'duorc_ParaphraseRC_answer_question', 
                        'duorc_ParaphraseRC_build_story_around_qa', 'duorc_ParaphraseRC_decide_worth_it', 'duorc_ParaphraseRC_extract_answer', 'duorc_ParaphraseRC_generate_question', 
                        'duorc_ParaphraseRC_generate_question_by_answer', 'duorc_ParaphraseRC_movie_director', 'duorc_ParaphraseRC_question_answering', 'duorc_ParaphraseRC_title_generation', 
                        'duorc_SelfRC_answer_question', 'duorc_SelfRC_build_story_around_qa', 'duorc_SelfRC_decide_worth_it', 'duorc_SelfRC_extract_answer', 'duorc_SelfRC_generate_question', 
                        'duorc_SelfRC_generate_question_by_answer', 'duorc_SelfRC_movie_director', 'duorc_SelfRC_question_answering', 'duorc_SelfRC_title_generation', 'gigaword_TLDR', 
                        'gigaword_first_sentence_title', 'gigaword_generate_summary_for_this', 'gigaword_in_a_nutshell', 'gigaword_make_a_title', 'gigaword_reverse_writing', 
                        'gigaword_write_a_title_for_this_sentence', 'gigaword_write_an_article', 'gigaword_write_its_sentence', 'glue_mrpc_equivalent', 'glue_mrpc_generate_paraphrase', 
                        'glue_mrpc_generate_sentence', 'glue_mrpc_paraphrase', 'glue_mrpc_replace', 'glue_mrpc_same_thing', 'glue_mrpc_want_to_know', 'glue_qqp_answer', 'glue_qqp_duplicate', 
                        'glue_qqp_duplicate_or_not', 'glue_qqp_meaning', 'glue_qqp_quora', 'glue_qqp_same_thing', 'imdb_Movie_Expressed_Sentiment', 'imdb_Movie_Expressed_Sentiment_2', 
                        'imdb_Negation_template_for_positive_and_negative', 'imdb_Reviewer_Enjoyment', 'imdb_Reviewer_Enjoyment_Yes_No', 'imdb_Reviewer_Expressed_Sentiment', 
                        'imdb_Reviewer_Opinion_bad_good_choices', 'imdb_Reviewer_Sentiment_Feeling', 'imdb_Sentiment_with_choices_', 'imdb_Text_Expressed_Sentiment', 
                        'imdb_Writer_Expressed_Sentiment', 'kilt_tasks_hotpotqa_combining_facts', 'kilt_tasks_hotpotqa_complex_question', 'kilt_tasks_hotpotqa_final_exam', 
                        'kilt_tasks_hotpotqa_formulate', 'kilt_tasks_hotpotqa_straighforward_qa', 'multi_news_distill', 'multi_news_expand_reverse_task_', 'multi_news_summarize', 
                        'multi_news_summary_scenario', 'multi_news_synthesize', 'multi_news_what_are_the_key_points', 'paws_labeled_final_Concatenation', 
                        'paws_labeled_final_Concatenation_no_label', 'paws_labeled_final_Meaning', 'paws_labeled_final_Meaning_no_label', 'paws_labeled_final_PAWS_ANLI_GPT3', 
                        'paws_labeled_final_PAWS_ANLI_GPT3_no_label', 'paws_labeled_final_Rewrite', 'paws_labeled_final_Rewrite_no_label', 'paws_labeled_final_context_question', 
                        'paws_labeled_final_context_question_no_label', 'paws_labeled_final_paraphrase_task', 'paws_labeled_final_task_description_no_label', 'qasc_is_correct_1', 
                        'qasc_is_correct_2', 'qasc_qa_with_combined_facts_1', 'qasc_qa_with_separated_facts_1', 'qasc_qa_with_separated_facts_2', 'qasc_qa_with_separated_facts_3', 
                        'qasc_qa_with_separated_facts_4', 'qasc_qa_with_separated_facts_5', 'quail_context_description_question_answer_id', 'quail_context_description_question_answer_text', 
                        'quail_context_description_question_text', 'quail_context_question_answer_description_id', 'quail_context_question_answer_description_text', 
                        'quail_context_question_description_answer_id', 'quail_context_question_description_answer_text', 'quail_context_question_description_text', 
                        'quail_description_context_question_answer_id', 'quail_description_context_question_answer_text', 'quail_description_context_question_text', 'quail_no_prompt_id', 
                        'quail_no_prompt_text', 'quarel_choose_between', 'quarel_do_not_use', 'quarel_heres_a_story', 'quarel_logic_test', 'quarel_testing_students', 
                        'quartz_answer_question_based_on', 'quartz_answer_question_below', 'quartz_given_the_fact_answer_the_q', 'quartz_having_read_above_passage', 
                        'quartz_paragraph_question_plain_concat', 'quartz_read_passage_below_choose', 'quartz_use_info_from_paragraph_question', 'quartz_use_info_from_question_paragraph', 
                        'quoref_Answer_Friend_Question', 'quoref_Answer_Question_Given_Context', 'quoref_Answer_Test', 'quoref_Context_Contains_Answer', 'quoref_Find_Answer', 
                        'quoref_Found_Context_Online', 'quoref_Given_Context_Answer_Question', 'quoref_Guess_Answer', 'quoref_Guess_Title_For_Context', 'quoref_Read_And_Extract_', 
                        'quoref_What_Is_The_Answer', 'ropes_background_new_situation_answer', 'ropes_background_situation_middle', 'ropes_given_background_situation', 
                        'ropes_new_situation_background_answer', 'ropes_plain_background_situation', 'ropes_plain_bottom_hint', 'ropes_plain_no_background', 'ropes_prompt_beginning', 
                        'ropes_prompt_bottom_hint_beginning', 'ropes_prompt_bottom_no_hint', 'ropes_prompt_mix', 'ropes_read_background_situation', 'rotten_tomatoes_Movie_Expressed_Sentiment', 
                        'rotten_tomatoes_Movie_Expressed_Sentiment_2', 'rotten_tomatoes_Reviewer_Enjoyment', 'rotten_tomatoes_Reviewer_Enjoyment_Yes_No', 
                        'rotten_tomatoes_Reviewer_Expressed_Sentiment', 'rotten_tomatoes_Reviewer_Opinion_bad_good_choices', 'rotten_tomatoes_Reviewer_Sentiment_Feeling', 
                        'rotten_tomatoes_Sentiment_with_choices_', 'rotten_tomatoes_Text_Expressed_Sentiment', 'rotten_tomatoes_Writer_Expressed_Sentiment', 
                        'samsum_Generate_a_summary_for_this_dialogue', 'samsum_Given_the_above_dialogue_write_a_summary', 'samsum_Sum_up_the_following_dialogue', 'samsum_Summarize_', 
                        'samsum_Summarize_this_dialogue_', 'samsum_To_sum_up_this_dialog', 'samsum_Write_a_dialogue_that_match_this_summary', 'sciq_Direct_Question', 
                        'sciq_Direct_Question_Closed_Book_', 'sciq_Multiple_Choice', 'sciq_Multiple_Choice_Closed_Book_', 'sciq_Multiple_Choice_Question_First', 
                        'social_i_qa_Check_if_a_random_answer_is_valid_or_not', 'social_i_qa_Generate_answer', 'social_i_qa_Generate_the_question_from_the_answer', 'social_i_qa_I_was_wondering', 
                        'social_i_qa_Show_choices_and_generate_answer', 'social_i_qa_Show_choices_and_generate_index', 'trec_fine_grained_ABBR', 'trec_fine_grained_ABBR_context_first', 
                        'trec_fine_grained_DESC', 'trec_fine_grained_DESC_context_first', 'trec_fine_grained_ENTY', 'trec_fine_grained_HUM', 'trec_fine_grained_HUM_context_first', 
                        'trec_fine_grained_LOC', 'trec_fine_grained_LOC_context_first', 'trec_fine_grained_NUM', 'trec_fine_grained_NUM_context_first', 'trec_fine_grained_open', 
                        'trec_fine_grained_open_context_first', 'trec_pick_the_best_descriptor', 'trec_trec1', 'trec_trec2', 'trec_what_category_best_describe', 'trec_which_category_best_describes', 
                        'wiki_bio_comprehension', 'wiki_bio_guess_person', 'wiki_bio_key_content', 'wiki_bio_what_content', 'wiki_bio_who', 'wiki_hop_original_choose_best_object_affirmative_1', 
                        'wiki_hop_original_choose_best_object_affirmative_2', 'wiki_hop_original_choose_best_object_affirmative_3', 'wiki_hop_original_choose_best_object_interrogative_1', 
                        'wiki_hop_original_choose_best_object_interrogative_2', 'wiki_hop_original_explain_relation', 'wiki_hop_original_generate_object', 'wiki_hop_original_generate_subject', 
                        'wiki_hop_original_generate_subject_and_object', 'wiki_qa_Decide_good_answer', 'wiki_qa_Direct_Answer_to_Question', 'wiki_qa_Generate_Question_from_Topic', 
                        'wiki_qa_Is_This_True_', 'wiki_qa_Jeopardy_style', 'wiki_qa_Topic_Prediction_Answer_Only', 'wiki_qa_Topic_Prediction_Question_Only', 
                        'wiki_qa_Topic_Prediction_Question_and_Answer_Pair', 'wiki_qa_automatic_system', 'wiki_qa_exercise', 'wiki_qa_found_on_google', 
                        'wiqa_does_the_supposed_perturbation_have_an_effect', 'wiqa_effect_with_label_answer', 'wiqa_effect_with_string_answer', 'wiqa_what_is_the_final_step_of_the_following_process', 
                        'wiqa_what_is_the_missing_first_step', 'wiqa_what_might_be_the_first_step_of_the_process', 'wiqa_what_might_be_the_last_step_of_the_process', 
                        'wiqa_which_of_the_following_is_the_supposed_perturbation', 'xsum_DOC_boils_down_to_simple_idea_that', 'xsum_DOC_given_above_write_one_sentence', 
                        'xsum_DOC_how_would_you_rephrase_few_words', 'xsum_DOC_tldr', 'xsum_DOC_write_summary_of_above', 'xsum_article_DOC_summary', 'xsum_college_roommate_asked_DOC_so_I_recap', 
                        'xsum_read_below_DOC_write_abstract', 'xsum_summarize_DOC', 'xsum_summarize_this_DOC_summary', 'yelp_review_full_based_on_that', 'yelp_review_full_format_rating', 
                        'yelp_review_full_format_score', 'yelp_review_full_format_star', 'yelp_review_full_on_a_scale', 'yelp_review_full_so_i_would', 'yelp_review_full_this_place']

old_ud_t0_prompts=[
    "cos_e_v1.11_description_question_option_text",
    "glue_mrpc_want_to_know",
    "glue_qqp_duplicate_or_not",
    "paws_labeled_final_task_description_no_label",
    "wiki_qa_found_on_google",
    "cosmos_qa_description_context_question_text",
    "dream_baseline",
    "qasc_qa_with_separated_facts_3",
    "quail_context_question_answer_description_text",
    "quarel_choose_between",
    "quartz_paragraph_question_plain_concat",
    "sciq_Direct_Question",
    "social_i_qa_I_was_wondering",
    "wiki_hop_original_choose_best_object_affirmative_3",
    "amazon_polarity_Is_this_review",
    "app_reviews_categorize_rating_using_review",
    "imdb_Movie_Expressed_Sentiment_2",
    "rotten_tomatoes_Text_Expressed_Sentiment",
    "yelp_review_full_this_place",
    "ag_news_classify_question_first",
    "dbpedia_14_given_a_choice_of_categories_",
    "trec_what_category_best_describe",
    "trec_fine_grained_open_context_first",
     #xhk: end of CLS. the followings are GEN
    "kilt_tasks_hotpotqa_combining_facts",
    "adversarial_qa_dbidaf_answer_the_following_q",
    "adversarial_qa_dbert_answer_the_following_q",
    "adversarial_qa_droberta_answer_the_following_q",
    "quoref_Answer_Question_Given_Context",
    "duorc_SelfRC_movie_director",
    "ropes_plain_background_situation",
    "wiqa_effect_with_string_answer",
    "common_gen_Put_together",
    "wiki_bio_key_content",
    "cnn_dailymail_3.0.0_news_card_view",
    "gigaword_generate_summary_for_this",
    "multi_news_summarize",
    "samsum_Generate_a_summary_for_this_dialogue",
    "xsum_DOC_how_would_you_rephrase_few_words",
    ]

our_new_prompts_615=[
    "cos_e_v1.11_description_question_option_text",
    "glue_mrpc_want_to_know",
    "glue_qqp_duplicate_or_not",
    "paws_labeled_final_task_description_no_label",
    "wiki_qa_found_on_google",
    "cosmos_qa_description_context_question_text",
    "dream_baseline",
    "qasc_qa_with_separated_facts_3",
    #"quail_context_question_answer_description_text",
    "quarel_choose_between",
    "quartz_paragraph_question_plain_concat",
    "sciq_Direct_Question", #sometimes long
    "social_i_qa_I_was_wondering",
    #"wiki_hop_original_choose_best_object_affirmative_3",
    "amazon_polarity_Is_this_review",
    "app_reviews_categorize_rating_using_review",
    "imdb_Movie_Expressed_Sentiment_2",
    "rotten_tomatoes_Text_Expressed_Sentiment",
    "yelp_review_full_this_place",
    "ag_news_classify_question_first",
    "dbpedia_14_given_a_choice_of_categories_",
    "trec_what_category_best_describe",
    ]

our_new_prompts_no_sentiment_topic_615=[
    "cos_e_v1.11_description_question_option_text",
    "glue_mrpc_want_to_know",
    "glue_qqp_duplicate_or_not",
    "paws_labeled_final_task_description_no_label",
    "wiki_qa_found_on_google",
    "cosmos_qa_description_context_question_text",
    "dream_baseline",
    "qasc_qa_with_separated_facts_3",
    #"quail_context_question_answer_description_text",
    "quarel_choose_between",
    "quartz_paragraph_question_plain_concat",
    "sciq_Direct_Question", #sometimes long
    "social_i_qa_I_was_wondering",
    #"wiki_hop_original_choose_best_object_affirmative_3",
    ]

our_prompts=old_ud_t0_prompts
#our_prompts=our_new_prompts_615
#our_prompts=our_new_prompts_no_sentiment_topic_615

max_choices=5
cut_length=50000
#cut_length=50
max_seq_length=256

#use_data_type="full"
use_data_type="cls"
cut_strategy=0
trec_fine_grained=False
root_file = "./huggingface_datasets" 
root_raw_file = "./huggingface_datasets" 

save_file = "./data_for_simcse" # where we save our modified data

#file_title=f"data614_{use_data_type}_new_{max_choices}choices_cut{int(cut_length/10000)}w_strategy{cut_strategy}_maxlen{max_seq_length}"
#file_title=f"data614_{use_data_type}_new_{max_choices}choices_cut{int(cut_length/10000)}w_strategy{cut_strategy}_no_sentiment_topic_maxlen{max_seq_length}"
#file_title=f"data616_{use_data_type}_old_{max_choices}choices_maxlen{max_seq_length}"
file_title=f"data725_{use_data_type}_old_{max_choices}choices_maxlen{max_seq_length}_unidir"
if "trec_fine_grained_open_context_first" in our_prompts:
    our_prompts.remove("trec_fine_grained_open_context_first")

write_path=os.path.join(save_file, file_title+".json")
datainfo_file=open(os.path.join(save_file,file_title+"_info"),'w')

print("file_title", file_title)

if use_data_type=="full":
    train_template_names=full_template_names
else:
    train_template_names=cls_template_names

train_string=[]
for task in train_task_family:
    for name in TASK_TYPE_DICT[task]:
        train_string.append(name)

train_datasets=[]
all_data=os.listdir(root_file)

num_prompt={}
for dataset_string in train_string:
    dataset_name=dataset_string.replace("/","_")
    cnt=0
    for name in train_template_names:
        if (dataset_name in name) and (name in all_data):
            cnt+=1
            train_datasets.append(name)
    for name in train_template_names:
        if dataset_name in name and name in all_data:
            num_prompt[name]=cnt


tokenizer = T5Tokenizer.from_pretrained("./huggingface_models/t5-large-lm-adapt")

print(train_datasets,"\n",len(train_datasets))

#train_datasets=["dream_baseline"]

data_count=0
long_data_count=0
expanded_data_count=0
new_data=[]

def token_cut(pre_input, cut_token_length=max_seq_length):
    global long_data_count

    pre_input=pre_input.strip()
    tokenized_pre_input = tokenizer(pre_input, padding=True, return_tensors="pt")
    len_pre_input=len(tokenized_pre_input["input_ids"][0])
    if(len_pre_input>cut_token_length):
        token_cut=tokenized_pre_input["input_ids"][0][-cut_token_length:-1]
        after_input=tokenizer.decode(token_cut)
        long_data_count+=1
        return after_input
    else:
        return pre_input

def clean_format(input):
    input=input.strip()
    while input.count("\n\n")>0:
        input=input.replace("\n\n","\n")
    while input.count("  ")>0:
        input=input.replace("  "," ")
    return input

data_stats=[]

choice_rng=random.Random(518)

def estimate_num_choices(dataset,data_id):
    cnt_choices=0
    for id in data_id:
        cnt_choices+=min(len(dataset[id]["answer_choices"]),max_choices)
    return int(cnt_choices/len(data_id))

for name in train_datasets:
    if name not in our_prompts:
        continue

    read_path = os.path.join(root_file,name,"train")
    print(f"Begin process:{read_path}")
    print(f"Begin process:{read_path}",file=datainfo_file)
    dataset=load_from_disk(read_path)

    is_raw_data=False

    if "quarel" in name:
        read_path = os.path.join(root_raw_file,name.split("_")[0],"train")
        dataset=load_from_disk(read_path)
        is_raw_data=True

    #print(dataset[0])
    #print(dataset[0],file=datainfo_file)

    if "answer_choices" in dataset[0] or is_raw_data==True: # if this dataset has choices
        #assert(len(dataset[0]["answer_choices"])<=max_choices)

        total=len(dataset)

        if is_raw_data==False:
            rng=random.Random(1234)
            avg_num_chocies=estimate_num_choices(dataset,rng.sample(population=list(range(total)),k=100))
        else:
            avg_num_chocies=2 # default
        
        if cut_strategy==0:
            if total<cut_length:
                cut_total=total
            else:
                cut_total=int(cut_length/num_prompt[name])
        elif cut_strategy==1:
            if total<cut_length:
                cut_total=total
            else:
                cut_total=cut_length
        elif cut_strategy==2:
            cut_total=min(int(cut_length*2/avg_num_chocies),total)

        print(name, "len:", len(dataset), "CLS" if "answer_choices" in dataset[0] else "GEN")
        print(name, "len:", len(dataset), "CLS" if "answer_choices" in dataset[0] else "GEN", file=datainfo_file)    

        rng=random.Random(1234)
        data_id=rng.sample(population=list(range(total)),k=cut_total)
        selected_dataset=dataset.select(data_id)
    
        data_count+=cut_total

        print("cut total", cut_total, "\n")
        print("cut total", cut_total, file=datainfo_file)

        cur_expanded_data_count=0
        for i in range(cut_total):
            example=selected_dataset[i]

            if is_raw_data==False:
                input=example["inputs_pretokenized"]
                target=example["targets_pretokenized"]
                choices=example["answer_choices"]

            need_choice_attached=False

            new_example={}
            if "cos_e_v1.11_description_question_option_text" in name:
                input=input.replace('Pick the option in line with common sense to answer the question.','')
                input=input.replace("Questions: ","")
                input=input.split("Options:")[0]
                pre_input=input
                need_choice_attached=True
            elif "glue_mrpc_want_to_know" in name:
                input2=input.split("I want to know whether the following two sentences mean the same thing.")[-1]
                input3=input2.split("Do they?")[0]
                new_example["sent"]=token_cut(input3)
                new_example["num_copies"]=1
                if "yes" in target:
                    new_example["label0"]=0
                    new_example["label1"]=1
                else:
                    new_example["label0"]=1
                    new_example["label1"]=0
                new_example["sent"]=clean_format(new_example["sent"])
                new_data.append(new_example)
                cur_expanded_data_count+=1
                if i<10:
                    print(new_example)
                    print(new_example, file=datainfo_file)
            elif "glue_qqp_duplicate_or_not" in name:
                input2=input.split("How is the life of a math student? Could you describe your own experiences?")[-1]
                input3=input2.split('Pick one: These questions are "duplicates" or "not duplicates".')[0]
                new_example["sent"]=token_cut(input3)
                new_example["num_copies"]=1
                if "not duplicates" in target:
                    new_example["label0"]=1
                    new_example["label1"]=0
                else:
                    new_example["label0"]=0
                    new_example["label1"]=1
                new_example["sent"]=clean_format(new_example["sent"])
                new_data.append(new_example)
                cur_expanded_data_count+=1
                if i<10:
                    print(new_example)
                    print(new_example, file=datainfo_file)
            elif "paws_labeled_final_task_description_no_label" in name:
                input2=input.split("Determine if the following two sentences paraphrase each other or not.")[-1]
                input3=input2.split("Sent 1: ")[-1]
                input4=input3.split('Sent 2: ')
                input5=input4[0]+input4[1]
                new_example["sent"]=token_cut(input5)
                new_example["num_copies"]=1
                if "Yes" in target:
                    new_example["label0"]=0
                    new_example["label1"]=1
                else:
                    new_example["label0"]=1
                    new_example["label1"]=0
                new_example["sent"]=clean_format(new_example["sent"])
                new_data.append(new_example)
                cur_expanded_data_count+=1
                if i<10:
                    print(new_example)
                    print(new_example, file=datainfo_file)
            elif "wiki_qa_found_on_google" in name:
                input2=input.split("Question: ")[-1]
                input3=input2.split("I found the following answer on Google: ")
                input4=input3[0]+input3[1]
                input5=input4.split("Is that a correct answer? Yes or no.")[0]
                new_example["sent"]=token_cut(input5)
                new_example["num_copies"]=1
                if "Yes" in target:
                    new_example["label0"]=0
                    new_example["label1"]=1
                else:
                    new_example["label0"]=1
                    new_example["label1"]=0
                new_example["sent"]=clean_format(new_example["sent"])
                new_data.append(new_example)
                cur_expanded_data_count+=1
                if i<10:
                    print(new_example)
                    print(new_example, file=datainfo_file)
            elif "cosmos_qa_description_context_question_text" in name:
                input2=input.split("Read the following context and answer the question.")[-1]
                input3=input2.split("Context: ")[-1]
                input4=input3.split("Question: ")
                input5=input4[0]+input4[1]
                input5=input5.replace("Answer:","")
                pre_input=input5
                need_choice_attached=True
            elif "dream_baseline" in name:
                input2=input.split("Dialogue:")[-1]
                input3=input2.split("Question: ")
                input4=input3[0]+input3[1]
                input5=input4.split("- ")[0]
                pre_input=input5
                need_choice_attached=True
            elif "qasc_qa_with_separated_facts_3" in name:
                input=input.replace("Fact 1: ","")
                input=input.replace("Fact 2: ","")
                input=input.replace("Given the two facts above, ","")
                pre_input=input
                need_choice_attached=True
            elif "quail_context_question_answer_description_text" in name:
                input=input.replace("Question: ","")
                input2=input.split("Options:")[0]
                pre_input=input2
                need_choice_attached=True
            elif "quarel_choose_between" in name:
                """input2=input.split("Question: ")[-1]
                input3=input2.split("(A)")[0]
                pre_input=input3
                need_choice_attached=True"""
                raw_input=example["question"].split("(")
                pre_input=raw_input[0]
                choices=[raw_input[1][3:],raw_input[2][3:]]
                target=choices[example["answer_index"]]
                need_choice_attached=True
            elif "quartz_paragraph_question_plain_concat" in name:
                pre_input=input
                need_choice_attached=True
            elif "sciq_Direct_Question" in name:
                input=input.replace("Answer the following question given this paragraph: ","")
                input=input.replace("Q: ","")
                input=input.replace("A:","")
                pre_input=input
                need_choice_attached=True
            elif "social_i_qa_I_was_wondering" in name:
                input=input.replace("I heard that ","")
                input=input.replace("And I was wondering ","")
                pre_input=input
                need_choice_attached=True
            elif "wiki_hop_original_choose_best_object_affirmative_3" in name:
                input=input.replace("Information:","")
                input=input.replace("After reading the paragraphs above, we are interested in knowing the entity with which '","")
                input=input.replace("' exhibits the relationship of '"," ") # may consider reverse the order
                input2=input.split("'. Find the answer from the choices below.")[0]
                pre_input=input2
                need_choice_attached=True
            elif "amazon_polarity_Is_this_review" in name:
                input=input.replace("Title: ","")
                input=input.replace("Review: ","")
                input=input.replace("Is the review positive or negative? ","")
                
                """pre_input=input
                choices=["Positive","Negative"]
                need_choice_attached=True"""

                input2=input+" Positive"
                new_example["sent"]=input2
                new_example["num_copies"]=1
                if "Positive" in target:
                    new_example["label0"]=0
                    new_example["label1"]=1
                else:
                    new_example["label0"]=1
                    new_example["label1"]=0
                new_example["sent"]=clean_format(new_example["sent"])
                new_data.append(new_example)
                cur_expanded_data_count+=1
                if i<10:
                    print(new_example)
                    print(new_example, file=datainfo_file)
            elif "app_reviews_categorize_rating_using_review" in name:
                input=input.replace('Given this review: "','')
                input=input.replace('"\nWould you recommend this app to a friend? Not at all, No, Maybe, Yes, or Definitely?','')
                #input2=input+" Definitely"
                input2=input+" Positive"
                new_example={}
                new_example["sent"]=input2
                new_example["num_copies"]=1
                if "Definitely" in target:
                    new_example["label0"]=0
                    new_example["label1"]=4
                elif "Yes" in target:
                    new_example["label0"]=1
                    new_example["label1"]=3
                elif "Maybe" in target:
                    new_example["label0"]=2
                    new_example["label1"]=2
                elif "No" in target:
                    new_example["label0"]=3
                    new_example["label1"]=1
                elif "Not at all" in target:
                    new_example["label0"]=4
                    new_example["label1"]=0
                new_example["sent"]=clean_format(new_example["sent"])
                new_data.append(new_example)
                cur_expanded_data_count+=1

                if i<10:
                    print(new_example)
                    print(new_example, file=datainfo_file)

                """input2=input+" Negative"
                new_example={}
                new_example["sent"]=input2
                new_example["num_copies"]=1
                if "Definitely" in target:
                    new_example["label0"]=4
                    new_example["label1"]=0
                elif "Yes" in target:
                    new_example["label0"]=3
                    new_example["label1"]=1
                elif "Maybe" in target:
                    new_example["label0"]=2
                    new_example["label1"]=2
                elif "No" in target:
                    new_example["label0"]=1
                    new_example["label1"]=3
                elif "Not at all" in target:
                    new_example["label0"]=0
                    new_example["label1"]=4
                new_example["sent"]=clean_format(new_example["sent"])
                new_data.append(new_example)
                cur_expanded_data_count+=1

                if i<10:
                    print(new_example)
                    print(new_example, file=datainfo_file)"""
            elif "imdb_Movie_Expressed_Sentiment_2" in name:
                input=input.replace("The following movie review expresses what sentiment? ","")
                """pre_input=input
                choices=["Positive","Negative"]
                target=target.strip().capitalize()
                need_choice_attached=True"""

                input2=input+" Positive"
                new_example["sent"]=input2
                new_example["num_copies"]=1
                if "Positive" in target:
                    new_example["label0"]=0
                    new_example["label1"]=1
                else:
                    new_example["label0"]=1
                    new_example["label1"]=0
                new_example["sent"]=clean_format(new_example["sent"])
                new_data.append(new_example)
                cur_expanded_data_count+=1
                if i<10:
                    print(new_example)
                    print(new_example, file=datainfo_file)
            elif "rotten_tomatoes_Text_Expressed_Sentiment" in name:
                input=input.replace(" What is the sentiment expressed in this text?","")
                """pre_input=input
                choices=["Positive","Negative"]
                target=target.strip().capitalize()
                need_choice_attached=True"""

                input2=input+" Positive"
                new_example["sent"]=input2
                new_example["num_copies"]=1
                if "Positive" in target:
                    new_example["label0"]=0
                    new_example["label1"]=1
                else:
                    new_example["label0"]=1
                    new_example["label1"]=0
                new_example["sent"]=clean_format(new_example["sent"])
                new_data.append(new_example)
                cur_expanded_data_count+=1
                if i<10:
                    print(new_example)
                    print(new_example, file=datainfo_file)
            elif "yelp_review_full_this_place" in name:
                input=input.replace("My rating for this place is ","")
                #input2=input+" 5 stars"
                input2=input+" Positive"
                new_example["sent"]=input2
                new_example["num_copies"]=1
                if "5 stars" in target:
                    new_example["label0"]=0
                    new_example["label1"]=4
                elif "4 stars" in target:
                    new_example["label0"]=1
                    new_example["label1"]=3
                elif "3 stars" in target:
                    new_example["label0"]=2
                    new_example["label1"]=2
                elif "2 stars" in target:
                    new_example["label0"]=3
                    new_example["label1"]=1
                elif "1 star" in target:
                    new_example["label0"]=4
                    new_example["label1"]=0
                new_example["sent"]=clean_format(new_example["sent"])
                new_data.append(new_example)
                cur_expanded_data_count+=1

                if i<10:
                    print(new_example)
                    print(new_example, file=datainfo_file)

                """input2=input+" Negative"
                new_example={}
                new_example["sent"]=input2
                new_example["num_copies"]=1
                if "5 stars" in target:
                    new_example["label0"]=4
                    new_example["label1"]=0
                elif "4 stars" in target:
                    new_example["label0"]=3
                    new_example["label1"]=1
                elif "3 stars" in target:
                    new_example["label0"]=2
                    new_example["label1"]=2
                elif "2 stars" in target:
                    new_example["label0"]=1
                    new_example["label1"]=3
                elif "1 star" in target:
                    new_example["label0"]=0
                    new_example["label1"]=4
                new_example["sent"]=clean_format(new_example["sent"])
                new_data.append(new_example)
                cur_expanded_data_count+=1

                if i<10:
                    print(new_example)
                    print(new_example, file=datainfo_file)"""
            elif "ag_news_classify_question_first" in name:
                input=input.replace("What label best describes this news article?","")
                pre_input=input
                need_choice_attached=True
            elif "dbpedia_14_given_a_choice_of_categories_" in name:
                input=input.replace(" Given a choice of categories company, educational institution, artist, athlete, office holder, mean of transportation, building, natural place, village, animal, plant, album, film or written work, the text refers to which one? ","")
                pre_input=input
                need_choice_attached=True
            elif "trec_what_category_best_describe" in name:
                input=input.replace("Categories: Description, Entity, Abbreviation, Person, Quantity, Location","")
                input=input.replace("What category best describes: ","")
                input=input.replace("Answer: ","")
                pre_input=input
                need_choice_attached=True
            elif "trec_fine_grained_ENTY" in name:
                input=input.replace("Is this question asking for an animal, an organ of the body, a color, creative piece, currency, disease or medicine, event, food, musical instrument, language, letter, plant, product, religion, sport, substance, symbol, technique, term, vehicle, word, other entity?","")
                pre_input=input
                need_choice_attached=True
            elif "trec_fine_grained_open_context_first" in name:
                input=input.replace("What is this question asking for?","")
                pre_input=input
                need_choice_attached=True
            elif "trec_fine_grained_LOC" in name:
                input=input.replace("Is this question asking for city, country, mountain, state, other location?","")
                pre_input=input
                need_choice_attached=True
            elif "trec_fine_grained_NUM" in name:
                input=input.replace("Is this question asking for code, count, date, distance, price, order, period of time, percentage, speed, temperature, size, weight, other number?","")
                pre_input=input
                need_choice_attached=True
            elif "trec_fine_grained_DESC" in name:
                input=input.replace("Is this question asking for definition, description, manner of action, reason?","")
                pre_input=input
                need_choice_attached=True
            elif "trec_fine_grained_ABBR" in name:
                input=input.replace("Is this question asking for an abbreviation, expression abbreviated?","")
                pre_input=input
                need_choice_attached=True
            elif "trec_fine_grained_HUM" in name:
                input=input.replace("Is this question asking for group, individual, title, description?","")
                pre_input=input
                need_choice_attached=True

            if need_choice_attached==True:
                pre_choices=[]
                for choice in choices:
                    pre_choices.append(choice)
                tokenized_pre_choices = tokenizer(pre_choices, padding=True, return_tensors="pt")

                #print(tokenized_pre_choices["input_ids"])

                max_len_pre_choices=0
                for index, choice in enumerate(choices):
                    max_len_pre_choices=max(max_len_pre_choices,len(tokenized_pre_choices["input_ids"][index]))     
                
                #print("max len pre choices",max_len_pre_choices)

                pre_input=token_cut(pre_input,max_seq_length-max_len_pre_choices-1)

                num_choices=0
                if len(choices)<=max_choices:
                    selected_choices=choices
                    num_choices=len(choices)
                else:
                    selected_choices=choice_rng.sample(population=choices,k=max_choices)
                    contain_target=False
                    for choice in selected_choices:
                        if choice.strip()==target.strip():
                            contain_target=True
                            break
                    if contain_target==False:
                        selected_choices=selected_choices[:-1]+[target]
                    num_choices=max_choices

                for choice in selected_choices:
                    if "_" in pre_input:
                        input=pre_input.replace("_",target.replace(target.strip(),choice.strip()))
                    elif "_____" in pre_input:
                        input=pre_input.replace("_____",target.replace(target.strip(),choice.strip()))
                    else:
                        input=pre_input+" "+target.replace(target.strip(),choice.strip())
                    new_example={}
                    #new_example["sent"]=token_cut(input)
                    new_example["sent"]=clean_format(input)
                    if choice.strip()==target.strip():
                        new_example["label0"]=0
                        new_example["label1"]=1
                        new_example["num_copies"]=1
                    else:
                        new_example["label0"]=1
                        new_example["label1"]=0
                        new_example["num_copies"]=num_choices-1
                    new_data.append(new_example)
                    if i<10:
                        print(new_example)
                        print(new_example, file=datainfo_file)
                cur_expanded_data_count+=num_choices
            
            """else:
                assert("sent" in new_example)
                assert("label0" in new_example)
                assert("label1" in new_example)
                assert("num_copies" in new_example)
                new_example["sent"]=clean_format(new_example["sent"])
                new_data.append(new_example)
                if i<2:
                    print(new_example)
                    print(new_example, file=datainfo_file)
                cur_expanded_data_count+=1"""
            
        
        expanded_data_count+=cur_expanded_data_count
        data_stats.append([name,"cls",cut_total,cur_expanded_data_count])


print("total datapoint count:", data_count)
#print("length<500 datapoint count:", data_remain_count)
print("expanded datapoint count:", expanded_data_count)
print("long datapoint count:", long_data_count)
print("data stats", data_stats)

print("total datapoint count:", data_count, file=datainfo_file)
#print("length<500 datapoint count:", data_remain_count)
print("expanded datapoint count:", expanded_data_count, file=datainfo_file)
print("long datapoint count:", long_data_count, file=datainfo_file)
print("data stats", data_stats, file=datainfo_file)

with open(write_path,"w") as jsonfile:
    for example in new_data:
        print(json.dumps(example),file=jsonfile)