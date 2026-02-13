from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    status: Mapped[str] = mapped_column(
        String, default="queued"
    )  # queued/running/finished/stopped/failed
    created_at: Mapped[int] = mapped_column(Integer)
    rules_yaml: Mapped[str] = mapped_column(Text)
    config_yaml: Mapped[str] = mapped_column(Text)
    progress_done: Mapped[int] = mapped_column(Integer, default=0)
    progress_total: Mapped[int] = mapped_column(Integer, default=0)
    summary_json: Mapped[str] = mapped_column(Text, default="{}")

    cases: Mapped[list["Case"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class Case(Base):
    __tablename__ = "cases"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(String, ForeignKey("runs.id"))
    index: Mapped[int] = mapped_column(Integer)

    prompt: Mapped[str] = mapped_column(Text)
    prompt_min: Mapped[str] = mapped_column(Text, default="")
    mutation: Mapped[str] = mapped_column(String, default="")

    response_text: Mapped[str] = mapped_column(Text, default="")
    trace_json: Mapped[str] = mapped_column(Text, default="{}")
    result_json: Mapped[str] = mapped_column(Text, default="{}")
    passed: Mapped[int] = mapped_column(Integer, default=0)

    run: Mapped["Run"] = relationship(back_populates="cases")


class WizardRun(Base):
    __tablename__ = "wizard_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    status: Mapped[str] = mapped_column(
        String, default="draft"
    )  # draft|discovering|configured|generating|ready|running|finished|failed|stopped
    endpoint: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[int] = mapped_column(Integer)
    progress_done: Mapped[int] = mapped_column(Integer, default=0)
    progress_total: Mapped[int] = mapped_column(Integer, default=0)

    agent_card_json: Mapped[str] = mapped_column(Text, default="{}")
    fuzzer_config_json: Mapped[str] = mapped_column(Text, default="{}")
    openai_config_json: Mapped[str] = mapped_column(Text, default="{}")
    rules_json: Mapped[str] = mapped_column(Text, default="{}")
    summary_json: Mapped[str] = mapped_column(Text, default="{}")

    testcases: Mapped[list["WizardTestcase"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    results: Mapped[list["WizardResult"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class WizardTestcase(Base):
    __tablename__ = "wizard_testcases"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(String, ForeignKey("wizard_runs.id"))
    prompt: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[int] = mapped_column(Integer)

    run: Mapped["WizardRun"] = relationship(back_populates="testcases")
    results: Mapped[list["WizardResult"]] = relationship(
        back_populates="testcase", cascade="all, delete-orphan"
    )


class WizardResult(Base):
    __tablename__ = "wizard_results"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(String, ForeignKey("wizard_runs.id"))
    testcase_id: Mapped[str] = mapped_column(String, ForeignKey("wizard_testcases.id"))

    response_text: Mapped[str] = mapped_column(Text, default="")
    trace_json: Mapped[str] = mapped_column(Text, default="{}")
    rule_results_json: Mapped[str] = mapped_column(Text, default="{}")
    passed: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[int] = mapped_column(Integer)

    run: Mapped["WizardRun"] = relationship(back_populates="results")
    testcase: Mapped["WizardTestcase"] = relationship(back_populates="results")
