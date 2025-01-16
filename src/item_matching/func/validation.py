from pydantic import BaseModel, computed_field, Field
from pathlib import Path
import polars as pl


class ValidationUserData(BaseModel):
    MATCH_BY: str = Field(default='text')
    FILE: Path
    MODE: str = Field(default='')
    MATCH_CATEGORY: str = Field(default='level3_global_be_category')
    existing_columns: set = Field(default=None)
    required_category: list = Field(default=None)

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
        self.required_category = category[:lv_index]

        required_columns += category
        if self.MATCH_BY == 'image':
            required_columns += ['image_url']
        return set(required_columns)

    @computed_field
    @property
    def read_data(self) -> str:
        file_type = self.FILE.suffix[1:]
        try:
            if file_type == 'csv':
                df = pl.read_csv(self.FILE)
            else:
                df = pl.read_parquet(self.FILE)
        except (pl.exceptions.ComputeError, FileNotFoundError) as e:
            return f"[UNKNOWN READING] **File {self.MODE}: {self.FILE.name}** file may be damaged: {e}"

        self.existing_columns = set(df.columns)

    @computed_field
    @property
    def validation(self) -> str:
        missing_columns = list(self.required_columns - self.existing_columns)
        if missing_columns:
            message = (
                f"[REQUIRED COLUMNS]: \n"
                f"Missing {missing_columns}. \n"
                f"Your columns: {list(self.existing_columns)} \n"
                f"If you choose **Match in which level category in {self.MATCH_CATEGORY}**. "
                f"Make sure to have {self.required_category} columns"
            )
            return message

    @computed_field
    @property
    def summary(self) -> str:
        pipe = [
            self.check_file_extension,
            self.read_data,
            self.validation,
        ]
        return '\n'.join([i for i in pipe if i])


# valid = ValidationUserData(
#     existing_columns={'item_name', 'level1_global_be_category'},
#     MATCH_BY='text',
#     FILE= Path('file.parquet'),
#     MODE='db',
#     MATCH_CATEGORY='level3_global_be_category',
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
