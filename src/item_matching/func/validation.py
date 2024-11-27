from pydantic import BaseModel, computed_field, Field
from typing import List, Any
from pathlib import Path
import duckdb
from rich import print


class ValidationUserData(BaseModel):
    MATCH_BY: str = Field(default='text')
    FILE: Path
    MODE: str = Field(default='')
    INPUT_COLUMNS: List[str] = Field(default=[])
    MATCH_CATEGORY: str = Field(default='level3_global_be_category')

    @computed_field
    @property
    def check_file_extension(self) -> str:
        if self.FILE.suffix not in ['.csv', '.parquet']:
            return '[FILE TYPE]: Please specify a .csv or .parquet file'

    @computed_field
    @property
    def required_columns(self) -> set:
        category = [
            'level1_global_be_category',
            'level2_global_be_category',
            'level3_global_be_category'
        ]
        required_columns = ['item_name']

        lv_index = int(self.MATCH_CATEGORY.split('_')[0][-1])
        category = category[:lv_index]

        required_columns += category
        if self.MATCH_BY == 'image':
            required_columns += ['image_url']
        return set(required_columns)

    @computed_field
    @property
    def read_data(self) -> str:
        try:
            data = duckdb.sql(f"select * from read_{self.FILE.suffix[1:]}('{self.FILE}') limit 10").pl()
            self.INPUT_COLUMNS = data.columns
        except Exception as errors:
            return f"[UNKNOWN READING] Your **{self.MODE}** file may be damaged: {errors}"

    @computed_field
    @property
    def validation(self) -> str:
        INPUT = set(self.INPUT_COLUMNS)
        missing_columns = list(self.required_columns - INPUT)
        if missing_columns:
            message = (
                f"[REQUIRED COLUMNS]: Missing {missing_columns}. "
                f"Your columns: {self.INPUT_COLUMNS}"
            )
            return message

    @computed_field
    @property
    def summary(self) -> str:
        pipe = [
            self.read_data,
            self.validation,
            self.check_file_extension,
        ]
        return '\n'.join([i for i in pipe if i])


# valid = UserData(
#     INPUT_COLUMNS=['item_name', 'level1_global_be_category', 'level2_global_be_category', 'level3_global_be_category'],
#     MATCH_BY='text',
#     FILE= Path('file.parquet'),
#     MODE='db'
# ).summary
# print(valid)
#
# valid = UserData(
#     INPUT_COLUMNS=['item_name', 'level1_global_be_category'],
#     MATCH_BY='image',
#     FILE= Path('file.parquet'),
#     MODE='db'
# ).summary
# print(valid)
