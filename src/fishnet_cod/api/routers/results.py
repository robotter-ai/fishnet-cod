from typing import Optional

from ..main import app
from ...core.model import Result


@app.get("/results/{result_id}")
async def get_result(result_id: str) -> Optional[Result]:
    return await Result.fetch(result_id).first()
