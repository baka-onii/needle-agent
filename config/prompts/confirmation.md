Review the tool selection before taking another action.
Is the highest-ranked available tool correct for the requested action? Inspect
both the tool's purpose and the proposed arguments; confidence alone is not
proof of correctness. If it is correct, explicitly select that tool by reissuing
ONE precise natural-language action beginning "Use TOOL_NAME to ..." inside a
<tool> block. Include exact paths and all literal values, putting bulk text in
payload blocks (<content> for one value, <text-1>, <text-2> for several). If it is incorrect, choose the appropriate
registered tool and provide a corrected atomic action instead. If the operation
cannot be performed safely or essential information is missing, explain that
with <final> or ask one genuinely necessary clarifying question.

Do not answer only "yes", silently retry the unchanged failed call, ask the human
to choose a tool, or invent arguments for an alternative candidate. This review
goes to the reasoning model, not ask_user. The chosen action is translated again
and must still clear validation, confidence, and safety. Approval and confidence
are independent; neither allows bypassing the other.
