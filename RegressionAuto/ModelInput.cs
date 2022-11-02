using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;
using System;

namespace RegressionAuto
{
    public class ModelInput
    {
        public ModelInput() {
        }

        public ModelInput(float year, float month, float day, float dayOfWeek, float hour, float timeOfDay, float holidays, float season, float weekend) {
            Year = year;
            Month = month;
            Day = day;
            DayOfWeek = dayOfWeek;
            Hour = hour;
            TimeOfDay = timeOfDay;
            Holidays = holidays;
            Season = season;
            WeekEnd = weekend;
        }

        public ModelInput(float year, float month, float day, float dayOfWeek, float hour,
            float timeOfDay, float holidays, float season, float weekend, float parking,
            float checkIns, float checkOuts, string date, float previous, float latest) {
            Year = year;
            Month = month;
            Day = day;
            DayOfWeek = dayOfWeek;
            Hour = hour;
            TimeOfDay = timeOfDay;
            Holidays = holidays;
            Season = season;
            WeekEnd = weekend;
            Parking = parking;
            CheckIns = checkIns;
            CheckOuts = checkOuts;
            Date = date;
            Previous = previous;
            Latest = latest;
        }


        [ColumnName(@"Year")]
        [LoadColumn(0)]
        public float Year { get; set; }

        [ColumnName(@"Month")]
        [LoadColumn(1)]
        public float Month { get; set; }

        [ColumnName(@"Day")]
        [LoadColumn(2)]
        public float Day { get; set; }

        [ColumnName(@"DayOfWeek")]
        [LoadColumn(3)]
        public float DayOfWeek { get; set; }

        [ColumnName(@"Hour")]
        [LoadColumn(4)]
        public float Hour { get; set; }

        [ColumnName(@"TimeOfDay")]
        [LoadColumn(5)]
        public float TimeOfDay { get; set; }

        [ColumnName(@"Holidays")]
        [LoadColumn(6)]
        public float Holidays { get; set; }

        [ColumnName(@"Season")]
        [LoadColumn(7)]
        public float Season { get; set; }

        [ColumnName(@"WeekEnd")]
        [LoadColumn(8)]
        public float WeekEnd { get; set; }

        [ColumnName(@"Parking")]
        [LoadColumn(9)]
        public float Parking { get; set; }

        [ColumnName(@"CheckIns")]
        [LoadColumn(10)]
        public float CheckIns { get; set; }

        [ColumnName(@"CheckOuts")]
        [LoadColumn(11)]
        public float CheckOuts { get; set; }

        [ColumnName(@"Date")]
        [LoadColumn(12)]
        public string Date { get; set; }

        [ColumnName(@"Previous")]
        [LoadColumn(13)]
        public float Previous { get; set; }
        [ColumnName(@"Latest")]
        [LoadColumn(14)]
        public float Latest { get; set; }

        public static void Main(string[] args) {
            ModelOutput modelOutput = new ModelOutput();
        }
    }

    public class ModelOutput
    {
        public float Score { get; set; }
    }
}
