from typing import Optional

from ...core.model import Result
from ..main import app


@app.get("/results/{result_id}")
async def get_result(result_id: str) -> Optional[Result]:
    return await Result.fetch(result_id).first()
