from pydantic import BaseModel, computed_field, Field
from pathlib import Path
import polars as pl
from typing import List, Set, Optional


class ValidationUserData(BaseModel):
    MATCH_BY: str = Field(default='text')
    FILE: Path
    MODE: str = Field(default='')
    MATCH_CATEGORY: str = Field(default='level3_global_be_category')
    existing_columns: Set[str] = Field(default_factory=set)
    default_columns: List[str] = Field(default=['item_name'])
    check_category: List[str] = Field(default_factory=list)
    external_data: bool = Field(default=False)

    @computed_field
    @property
    def _validate_file(self) -> Optional[str]:
        if self.FILE.suffix not in {'.csv', '.parquet'}:
            return '[FILE TYPE]: Please specify a .csv or .parquet file'

    @computed_field
    @property
    def required_columns(self) -> Set[str]:
        categories = [
            'level1_global_be_category',
            'level2_global_be_category',
            'level3_global_be_category'
        ]

        level = int(self.MATCH_CATEGORY.split('_')[0][-1])
        self.check_category = categories[:level]

        columns = self.default_columns.copy()
        if not self.external_data:
            columns.extend(self.check_category)
        if self.MATCH_BY == 'image':
            columns.append('image_url')

        return set(columns)

    @computed_field
    @property
    def _read_data(self) -> Optional[str]:
        try:
            df = (
                pl.read_csv(self.FILE)
                if self.FILE.suffix == '.csv'
                else pl.read_parquet(self.FILE).head(100)
            )
            df.write_parquet(self.FILE.parent / f'{self.MODE}.parquet')
            self.existing_columns = set(df.columns)
        except Exception as e:
            return f"[UNKNOWN READING] **File {self.MODE}: {self.FILE.name}** file may be damaged: {e}"

    @computed_field
    @property
    def _validate_columns(self) -> Optional[str]:
        missing = self.required_columns - self.existing_columns
        if missing:
            return (
                f"[REQUIRED COLUMNS]:\n"
                f"Missing {list(missing)}.\n"
                f"Your columns: {list(self.existing_columns)}\n"
                f"If you choose **Match in which level category in {self.MATCH_CATEGORY}**. "
                f"Make sure to have {self.check_category} columns"
            )

    @computed_field
    @property
    def summary(self) -> str:
        pipe = [
            self._validate_file,
            self._read_data,
            self._validate_columns,
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

valid = ValidationUserData(
    MATCH_BY='image',
    FILE= Path('/media/kevin/data_4t/test.parquet'),
    MODE='db'
).summary
print(valid)

valid = ValidationUserData(
    MATCH_BY='text',
    FILE= Path('/media/kevin/data_4t/test.parquet'),
    MODE='db',
    external_data=True
).summary
print(valid)
