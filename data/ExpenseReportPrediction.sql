SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
DROP TABLE IF EXISTS dbo.ExpenseReportPrediction;
GO
CREATE TABLE [dbo].[ExpenseReportPrediction](
    [ExpenseReportPredictionID] [int] IDENTITY(1,1) NOT NULL,
    [EmployeeID] [int] NOT NULL,
    [EmployeeName] [nvarchar](50) NOT NULL,
    [ExpenseCategoryID] [tinyint] NOT NULL,
    [ExpenseCategory] [nvarchar](50) NOT NULL,
    [ExpenseDate] [date] NOT NULL,
    [ExpenseYear] [int] NOT NULL,
    [Amount] [decimal](5, 2) NOT NULL,
    [PredictedAmount] [decimal](5, 2) NOT NULL
) ON [PRIMARY]
GO
ALTER TABLE [dbo].[ExpenseReportPrediction] ADD  CONSTRAINT [PK_ExpenseReportPrediction] PRIMARY KEY CLUSTERED 
(
    [ExpenseReportPredictionID] ASC
)WITH (STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ONLINE = OFF, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO