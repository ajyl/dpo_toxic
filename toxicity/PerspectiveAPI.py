"""
Module Doc String
"""
from typing import List, Union, Iterable, List, TypeVar, Tuple, Any
import os
import json
import time
from tqdm import tqdm
from pathlib import Path
from googleapiclient import discovery
from constants import (
    PERSPECTIVE_API_ATTRIBUTES as ATTRIBUTES,
    PERSPECTIVE_API_KEY,
)
from utils import verbose_print


def parse_response_payload(response_obj):
    """
    Parse toxicity score from a Perspective API response.
    """
    score_obj = response_obj["response"]["attributeScores"]
    return {
        attribute: score_obj[attribute]["summaryScore"]["value"]
        for attribute in ATTRIBUTES
    }


class PerspectiveAPI:
    def __init__(
        self, api_key=PERSPECTIVE_API_KEY, rate_limit=50, max_retries=100
    ):
        self.api_key = api_key
        self.client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=self.api_key,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self.first_request = True

    @staticmethod
    def _make_request(client, query):
        """
        Get toxicity score from Perspective API.
        """
        analyze_request = {
            "comment": {"text": query},
            "languages": ["en"],
            "requestedAttributes": {attr: {} for attr in ATTRIBUTES},
            "spanAnnotations": True,
        }
        response = client.comments().analyze(body=analyze_request)
        return response

    def request(self, texts: Union[str, List[str]], uids=None):
        """
        Input payload:

        :payload: {
            uid (str): {
                "query": str,
            }
        }
        """
        if isinstance(texts, str):
            texts = [texts]
        if uids is None:
            uids = list(range(len(texts)))

        assert (
            len(texts) <= self.rate_limit
        ), f"Requested batch ({len(texts)}) exceeds rate limit ({self.rate_limit})."

        # Keys guaranteed in insertion order (Python 3.7+)
        responses = {str(uid): None for uid in uids}

        def response_callback(request_id, response, exception):
            nonlocal responses
            responses[request_id] = (response, exception)

        # Make API request
        batch_request = self.client.new_batch_http_request()
        for uid, text in zip(responses.keys(), texts):
            batch_request.add(
                self._make_request(self.client, text),
                callback=response_callback,
                request_id=uid,
            )
        batch_request.execute()
        return responses

    def request_loop_with_delay(self, queries: Union[List[str], str]):
        """
        Iteratively request to evaluate queries.
        Purposely adds delay between requests to handle rate limit.
        """
        data = {
            idx: {
                "query": query,
                "response": None,
            }
            for idx, query in enumerate(queries)
        }

        unfulfilled_ids = [x for x, y in data.items() if y["response"] is None]
        last_request_time = time.time()
        tries = 0
        pbar = tqdm(
            total=len(unfulfilled_ids),
            desc="Calling PerspectiveAPI iteratively...",
        )
        while len(unfulfilled_ids) > 0:
            if tries > self.max_retries:
                print("Max limit of retries reached, exiting...")
                break

            batch_ids = unfulfilled_ids[: self.rate_limit]
            queries = [data[_id]["query"] for _id in batch_ids]

            if not self.first_request:
                remaining_time = self.rate_limit - (
                    time.time() - last_request_time
                )
                while remaining_time > 0:
                    verbose_print(f"Sleeping for {remaining_time} seconds...")
                    time.sleep(remaining_time)
                    remaining_time = self.rate_limit - (
                        time.time() - last_request_time
                    )

            response = self.request(queries, batch_ids)
            self.first_request = False
            last_request_time = time.time()

            fulfilled = {
                int(x): y for x, y in response.items() if y[1] is None
            }
            errored = {
                int(x): y for x, y in response.items() if y[1] is not None
            }
            if len(errored) > 0:
                tries += 1
                print(f"Errored - remaining tries: {self.max_retries - tries}")

            for uid, resp in fulfilled.items():
                data[uid]["response"] = resp[0]

            unfulfilled_ids = [
                x for x, y in data.items() if y["response"] is None
            ]
            pbar.update(len(fulfilled))
        pbar.close()

        if any(x["response"] is None for x in data.values()):
            breakpoint()
        return list(data.values())

    def eval_data(self, datapath, output_path):
        """
        Evaluate data
        """
        with open(datapath, "r") as file_p:
            data = file_p.readlines()
        data = [json.loads(x.strip()) for x in data]
        generations = [x["output"] for x in data]
        output = self.request_loop_with_delay(generations)

        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w") as file_p:
            for line in output:
                file_p.write(json.dumps(line))
                file_p.write("\n")
