import json
from strata.utils.logging import logger


def gt_to_response(gt):
    return json.dumps({key: gt[key] for key in gt})


def parse_response(response, response_start, verbose=False):
    response = response.replace(response_start, "")
    logger.debug(f"\nRaw response: {response}\n")

    try:
        response = json.loads(response)
    except json.decoder.JSONDecodeError:
        response = {}

    logger.debug(f"\nAnswers: {response}\n")
    return response
