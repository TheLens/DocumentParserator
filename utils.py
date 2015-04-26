import json
import re
import sys
import logging
import importlib
from documentparserator.settings import Settings
from documentcloud import DocumentCloud

SETTINGS = Settings()
module = importlib.import_module('documentparserator.parserator.contract_parser')
logging.basicConfig(level=logging.DEBUG, filename=SETTINGS.LOG_LOCATION)
client = DocumentCloud()


def get_colors(tag):
    """
    Get the colors for a certain tag
    """
    with open(SETTINGS.TAGS_LOCATION) as data_file:
        data = json.load(data_file)
    return [d for d in data if d['name']==tag].pop()


def sort_keys(keys):
    """
    Keys come in the form page#-tokennumber (on a page)
    Sort them in order of pages, then in order of token #
    """
    keys.sort(key=lambda x: int(x.split("-")[1]))
    keys.sort(key=lambda x: int(x.split("-")[0]))
    return keys
  

def get_document_page(doc_cloud_id, page):
    """
    Get a page in a document cloud document
    """
    doc = client.documents.get(doc_cloud_id)
    page_text = doc.get_page_text(page)
    page_text = page_text.decode("ascii", "ignore").encode("ascii", "ignore")
    return page_text


def get_labeled_tokens(doc_cloud_id, page):
    """
    Take a page of document cloud and return labeled tokens
    """
    page_text = get_document_page(doc_cloud_id, page_text)
    page_tokens =  module.tokenize(page_text)
    ids = {}
    counter = 1
    for t in tokens:
        tokenid = str(page) + "-" + str(counter)
        counter += 1
        output = {}
        output['page'] = page
        output['word'] = t
        output['count'] = counter
        ids[tokenid] = output
    return ids


def span_wrap(text, span_id, tag):
    if tag=="skip":
        return "<span id=\"" + span_id + "\" class=\"token\" data-tag=\"" + tag + "\">" + text + "</span>"
    else:
        colors = get_colors(tag)
        style = 'style="border: 2px solid rgb(' + str(colors['red']) + ',' +\
            ' ' + str(colors['green']) + ', ' + str(colors['blue']) + ');"'
        return "<span id=\"" + span_id +\
               "\" class=\"token\" data-tag=\"" +\
               tag + "\"" + style +\
               ">" + text + "</span>"

def spanify(page_text, page_no, labels=None):
    tokens = module.tokenize(page_text, True)
    last_index_mem = 0
    in_between = ""
    new_tokens = []
    in_betweens = []
    token_no = 1
    for t in tokens:
        start = t[0]
        end = t[1]
        token_no = token_no + 1
        if (last_index_mem > 0): 
            in_between = page_text[last_index_mem: start];
        last_index_mem = end;
        spanid = str(page_no) + "-" + str(token_no)
        if labels:
            try:
                correct_label = [l for l in labels if spanid==l['id']].pop()['label']
            except IndexError:
                logging.debug("Skipping. Could not find label for " + spanid)
                correct_label = "skip"
            new_token = span_wrap(str(page_text[start: end]), spanid, correct_label)
        else:
            new_token = span_wrap(str(page_text[start: end]), spanid, "skip")
        new_tokens.append(new_token)
        in_betweens.append(in_between)

    output = ""

    for i in range(0,len(new_tokens)):
        output = output + new_tokens[i] + in_betweens[i]

    return output.replace("\n", "<br />")