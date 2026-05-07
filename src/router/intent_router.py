
from src.schemas import State
from langchain.messages import HumanMessage, SystemMessage
from src.ai.model import llm
from src.schemas.model import Route


def llm_call_router(state: State):
    """Route the input to the appropriate node"""

    router = llm.with_structured_output(Route)

    # Run the augmented LLM with structured output to serve as routing logic
    decision = router.invoke(
        [
            SystemMessage(
                content="Route the input to the appropriate node based on the user's request."
            ),
            HumanMessage(content=state["input"]),
        ]
    )

    return {"decision": decision.step}


def route_decision(state: State):
    # Return the node name you want to visit next
    if  'SALES_RECO_NODE' == state['decision']:
        return 'SALES_RECO_NODE'
    elif 'COMPLIANCE_CHECK_NODE' == state['decision']:
        return 'COMPLIANCE_CHECK_NODE'
    elif 'VENDOR_ONBOARDING_NODE' == state['decision']:
        return 'VENDOR_ONBOARDING_NODE'
    elif 'OPS_STOCK_NODE' == state['decision']:
        return 'OPS_STOCK_NODE'
    elif 'GENERAL_KB_NODE' == state['decision']:
        return 'GENERAL_KB_NODE'
    else:
        return 'DEFAULT_NODE'