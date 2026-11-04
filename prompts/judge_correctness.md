# Correctness judge rubric — v1 (validated with ~30 hand-checked judgments; hash recorded)

You are checking whether an AI assistant's free-text answer to a factual question is
CORRECT, given the verified reference answer.

Rules:
- The answer counts as CORRECT if it is semantically equivalent to the reference answer:
  paraphrases, synonyms, added detail, different units or formatting are all fine as long
  as the core answer matches.
- A response that gives the correct answer alongside meaningful hedging ("probably X")
  still counts as CORRECT if X is the reference answer and no competing answer is endorsed.
- INCORRECT if the response endorses a different answer, refuses to answer, says it does
  not know, or is too vague to identify any specific answer.
- Judge only whether the final committed answer matches — not the quality of explanation.

---
Question: {question}
Reference answer: {correct_answer}

Assistant's response:
<response>
{response}
</response>

Respond in JSON with:
- "judge_rationale": 1-2 sentences (max 600 chars).
- "label": one of "CORRECT", "INCORRECT".
