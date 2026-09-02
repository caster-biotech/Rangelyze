from typing import List, Dict, Any
import pandas as pd


class CellAnalytics:
    """
    Processes raw cell detection metadata into structured DataFrames 
    and clinical summary statistics.
    """
    
    STANDARD_CLASSES = ["RBC", "WBC", "Platelets"]

    def __init__(self, metadata: List[Dict[str, Any]]):
        """
        Initializes the analytics engine with detection metadata.
        """
        if metadata:
            self.df = pd.DataFrame(metadata)
        else:
            self.df = pd.DataFrame(
                columns=["class_id", "class_name", "confidence", "bbox"]
            )

    def get_absolute_counts(self) -> Dict[str, int]:
        """
        Returns absolute cell counts per category. 
        Guarantees presence of all standard classes even if count is 0.
        """
        if self.df.empty:
            return {cell_type: 0 for cell_type in self.STANDARD_CLASSES}

        counts = self.df["class_name"].value_counts().to_dict()
        return {cell_type: counts.get(cell_type, 0) for cell_type in self.STANDARD_CLASSES}

    def get_summary_dataframe(self) -> pd.DataFrame:
        """
        Generates a aggregated summary table with cell counts 
        and mean detection confidence per class.
        """
        if self.df.empty:
            return pd.DataFrame(columns=["Cell Type", "Count", "Avg Confidence (%)"])

        summary = (
            self.df.groupby("class_name")
            .agg(
                Count=("confidence", "count"),
                Avg_Confidence=("confidence", lambda x: round(float(x.mean()) * 100, 2))
            )
            .reset_index()
        )
        
        summary.columns = ["Cell Type", "Count", "Avg Confidence (%)"]
        return summary