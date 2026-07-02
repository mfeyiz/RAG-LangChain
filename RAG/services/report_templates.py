"""Starter Markdown for newly-created reports in the Create/Edit studio.

A new report is just a workspace Markdown source (`<stem>.md`) seeded with one of
these skeletons. ``{title}`` is substituted with the user-supplied report title so
the document already carries a first heading (which also guarantees at least one
chunk, so the report shows up in ``list_documents("workspace")``).
"""

_BLANK = """# {title}

"""

_BUSINESS_REPORT = """# {title}

## Executive Summary

_Summarize the key findings and recommendations here._

## Background

_Describe the context and objectives of this report._

## Analysis

_Present your analysis. Insert charts and tables from the tools panel._

## Key Metrics

| Metric | Value | Change |
| --- | --- | --- |
|  |  |  |

## Recommendations

- _Recommendation one_
- _Recommendation two_

## Conclusion

_Wrap up the report._
"""

_RESEARCH_SUMMARY = """# {title}

## Abstract

_A short summary of the research question and findings._

## Question

_What are we trying to answer?_

## Findings

_Present the evidence. Drag citations from the chat into this section._

## Discussion

_Interpret the findings and note limitations._

## References

1. _Source one_
"""

_PROJECT_STATUS = """# {title}

## Status Overview

**Overall status:** 🟢 On track

## Progress This Period

- _What was completed_

## Upcoming

- _What is planned next_

## Risks & Blockers

| Risk | Impact | Mitigation |
| --- | --- | --- |
|  |  |  |

## Metrics

_Insert a chart of the relevant metrics from the tools panel._
"""

TEMPLATES = {
    "blank": _BLANK,
    "business-report": _BUSINESS_REPORT,
    "research-summary": _RESEARCH_SUMMARY,
    "project-status": _PROJECT_STATUS,
}


def render(template: str, title: str) -> str:
    """Return the seed Markdown for ``template`` with ``title`` substituted.

    Falls back to the blank template for an unknown key.
    """
    skeleton = TEMPLATES.get(template, _BLANK)
    safe_title = (title or "Untitled report").strip() or "Untitled report"
    return skeleton.format(title=safe_title)
