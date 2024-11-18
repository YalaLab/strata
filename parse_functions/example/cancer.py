import json
from strata.utils.logging import logger


def gt_to_response(gt):
    assert len(gt) == 2
    return json.dumps({"left": int(gt["Cancer_LB"]), "right": int(gt["Cancer_RB"])})


def parse_response(response, response_start, verbose=False):
    response = response.replace(response_start, "").lower()
    logger.debug(f"\nRaw response: {response}\n")

    response = json.loads(response)

    # Note that according to our prompt in `templates.yaml`, the model could also predict -1 if the tissue is not mentioned. In this example, we will consider it as 0. You can also handle it differently according to your use case.
    answers = {
        "Cancer_LB": 1 if response["left"] == 1 else 0,
        "Cancer_RB": 1 if response["right"] == 1 else 0,
    }
    logger.debug(f"\nAnswers: {response}\n")
    return answers
