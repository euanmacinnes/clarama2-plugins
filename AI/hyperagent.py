"""
Hyperagent state machine and model for multi-bot Q&A flows.

Defines a MultiGenieModel and a MultiGenieMachine orchestrating multiple
specialist bots, rephrasing questions, producing answers, and optional news
searches.
"""
import json
from typing import Optional

from genie_flow.genie import GenieModel, GenieStateMachine
from genie_flow.model.template import MapTaskTemplate
from loguru import logger
from pydantic import Field
from statemachine import State
from statemachine.event_data import EventData


# Hyperagent
#   Task agent
#   Help agent
#   Code Agent
#   Analytical Agent (dataframe focussed)
#   Unit Operation Agent


class MultiGenieModel(GenieModel):
    """
    Pydantic model for the multi-agent hyperagent.

    Stores the original user question, per-bot rephrasings and answers,
    and optional news queries/results.
    """
    original_question: str = Field("", description="Original question asked")
    bots: dict[str, str] = Field(
        default={
            "Task Dictionary": "task_dictionary/task_bot.jinja2",
        },
        description="dictionary mapping bot names to their descriptive templates",
    )
    rephrased_questions: list[dict[str, str]] = Field(
        default_factory=list,
        description="Rephrased questions per bot",
    )
    answer: str = Field(
        default="",
        description="Answers per bot",
    )
    follow_up: str = Field(
        default="",
        description="Request for follow-up",
    )
    news_queries: list[str] = Field(
        default_factory=list,
        description="List of news queries",
    )
    news: list[dict[str, Optional[str]]] = Field(
        default_factory=list,
        description="List of news items found",
    )

    @classmethod
    def get_state_machine_class(cls) -> type["GenieStateMachine"]:
        return MultiGenieMachine


class MultiGenieMachine(GenieStateMachine):
    """
    State machine orchestrating the multi-bot flow.

    States include intro, deciding paths, creating answers and news, and
    presenting results. Template mapping defines prompts/responses for steps.
    """
    intro = State(initial=True, value=0, exit="capture_original_question")

    # initial state calls intro first.

    ai_decides_path = State(value=100)
    user_views_rephrasing = State(value=110)
    user_views_no_bots_found = State(value=120, exit="capture_original_question")

    ai_creates_answer = State(value=200, exit="capture_answer")
    ai_creates_news_queries = State(value=210, exit="capture_news_queries")
    searching_news = State(value=220, exit="capture_news")
    user_views_answer = State(value=230, exit="capture_original_question")

    user_input = (
            intro.to(ai_decides_path)
            | user_views_answer.to(ai_decides_path)
            | user_views_no_bots_found.to(ai_decides_path)
    )

    processing_done = (
            ai_decides_path.to(user_views_rephrasing, cond="bots_found")
            | ai_decides_path.to(user_views_no_bots_found, unless="bots_found")
            | ai_creates_answer.to(ai_creates_news_queries)
            | ai_creates_news_queries.to(searching_news)
            | searching_news.to(user_views_answer)
    )

    advance = user_views_rephrasing.to(ai_creates_answer)

    templates = dict(
        intro="response/intro.jinja2",  # Response is the "statically rendered content"
        ai_decides_path="hyperagent/choose_bot.jinja2",
        user_views_rephrasing="response/user_views_rephrasing.jinja2",
        user_views_no_bots_found="response/no_bots.jinja2",
        ai_creates_answer=[
            MapTaskTemplate(
                "llm_answer/ai_answers_question.jinja2",
                "rephrased_questions[*]",
            ),
            dict(
                answer="llm_answer/ai_consolidates_answers.jinja2",
                follow_up="llm_converse/ai_ask_followup.jinja2",
            )
        ],
        ai_creates_news_queries="llm_converse/ai_creates_news_queries.jinja2",
        searching_news=MapTaskTemplate(
            "web_search/search_news.jinja2",
            "news_queries[*]",
        ),
        user_views_answer="response/user_views_answer.jinja2",
    )

    def capture_original_question(self):
        self.model.original_question = self.model.actor_input

    def capture_answer(self):
        try:
            answer_dict = json.loads(self.model.actor_input)
        except json.decoder.JSONDecodeError:
            return
        self.model.answer = answer_dict["answer"]
        self.model.follow_up = answer_dict["follow_up"]

    def capture_news_queries(self):
        queries = [
            q.replace('"', "")
            for q in self.model.actor_input.split("\n")
        ]
        logger.info(queries)
        self.model.news_queries = queries

    def capture_news(self):
        try:
            news_results = json.loads(self.model.actor_input)
        except json.decoder.JSONDecodeError:
            return

        news = [
            {
                k: n["results"][0][k]
                for k in ["title", "url", "content", "publishedDate"]
                if k in n["results"][0]
            }
            for n in news_results
            if len(n["results"]) > 0
        ]
        logger.info(news)
        self.model.news = news

    def on_exit_ai_decides_path(self):
        try:
            paths = json.loads(self.model.actor_input)
        except json.JSONDecodeError:
            logger.error("Invalid json in ai_decides_path")
            paths = dict()

        self.model.rephrased_questions = [
            {
                "name": bot_name,
                "template": self.model.bots.get(bot_name, "bots/unknown_bot.jinja2"),
                "question": bot_value["question"],
                "reason": bot_value["reason"],
            }
            for bot_name, bot_value in paths.items()
            if bot_name in self.model.bots.keys() and bot_value and bot_value["question"]
        ]

    def bots_found(self, event_data: EventData):
        try:
            print("EVENT DATA:", str(event_data))
            print("EVENT DATA ARGS:", str(event_data.args[0]))
            response = json.loads(event_data.args[0])
        except Exception:
            return False

        print("EVENT BOTS: ", str(self.model.bots.keys()))

        for bot_name, bot_value in response.items():
            if bot_name in self.model.bots.keys() and bot_value and bot_value["question"]:
                return True
        return False
