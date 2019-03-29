using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using CsvHelper;
using CsvHelper.Configuration;

namespace DoneDoneML
{
    public class Issue
    {
        public string Title {get;set;}
        public string Description {get;set;}
        public string Status {get;set;}
        public string FixerName {get;set;}
        public string FixerEmail {get;set;}

        public static IEnumerable<Issue> GetAllIssues(string path)
        {
            using(var reader = new CsvReader(File.OpenText(path)))
            {
                return reader.GetRecords<Issue>().ToList();
            }
        }
    }
}