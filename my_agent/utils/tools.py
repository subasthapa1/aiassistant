from langchain_core.tools import tool

@tool
def fetch_email() -> int:
    """Fetch email from the mail.

    Args:
        from: Sender's email
        subject: Title of an email
        date: Email received date
        message: Email's content

    """
    return true


@tool
def create_draft_email(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b


@tool
def summarize_email() -> float:
    """Read all emails received on a specific time frame and summarize them.

    Args:
        a: first int
        b: second int
    """
    return a / b
