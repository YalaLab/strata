import json
from strata.utils.logging import logger


def gt_to_response(gt):
    assert len(gt) == 2
    return json.dumps({"left": int(gt["Source_LB"]), "right": int(gt["Source_RB"])})


def parse_response(response, response_start, verbose=False):
    response = response.replace(response_start, "").lower()
    logger.debug(f"\nRaw response: {response}\n")

    response = json.loads(response)
    answers = {
        "Source_LB": 1 if response["left"] else 0,
        "Source_RB": 1 if response["right"] else 0,
    }
    logger.debug(f"\nAnswers: {response}\n")
    return answers
