create database if not exists salesDataWalmart ;
use salesDataWalmart ;
select * from `walmartsalesdata.csv`;
RENAME TABLE `walmartsalesdata.csv` TO sales;
select * from sales ;
-- add time of day column
select Time, (case when 'time' between "00:00:00" and "12:00:00" then "Morning" 
                   when 'time' between "12:01:00" and "16:00:00" then "Afternoon" 
                   else "Evening" end) as time_of_day from sales ;
SELECT
	time,
	(CASE
		WHEN `time` BETWEEN "00:00:00" AND "12:00:00" THEN "Morning"
        WHEN `time` BETWEEN "12:01:00" AND "16:00:00" THEN "Afternoon"
        ELSE "Evening"
    END) AS time_of_day
FROM sales;
alter table sales add column time_of_day varchar(20) ;
update sales set time_of_day = (CASE WHEN `time` BETWEEN "00:00:00" AND "12:00:00" THEN "Morning" WHEN `time` BETWEEN "12:01:00" AND "16:00:00" THEN "Afternoon" ELSE "Evening" END);
select date, dayname(date) from sales;
-- add day name column
alter table sales add column day_name varchar(10);
update sales set day_name = (dayname(date)) ;
select * from sales ;
-- add month name column
select date, monthname(date) from sales;
alter table sales add column month_name varchar(15);
update sales set month_name = (monthname(date));
-- -- How many unique cities does the data have?
select count(distinct city) from sales ;
-- --  In which city is each branch?
select distinct branch,city from sales;
---- How many unique product lines does the data have?
select count(distinct `Product line`) from sales ;
---- What is the most selling product line
select `Product line`,sum(Quantity) as qty from sales group by `Product line` order by qty desc ;
-- -- What is the most selling product line
select `Product line`,sum(Quantity) as qty from sales group by `Product line` order by qty desc ;
-- -- What is the total revenue by month
select distinct month_name as month, sum(Total) as total_revenue from sales group by month order by total_revenue;
-- -- which is the most common payment method
select Payment, count(Payment) as cnt from sales group by Payment order by cnt desc ;
-- -- What month had the largest COGS?
select  sum(cogs) as high_cogs, month_name from sales group by month_name order by high_cogs desc ;
-- -- What product line had the largest revenue?
select distinct `Product line`, sum(Total) as total_revenue from sales group by `Product line` order by total_revenue desc ;
-- -- What is the city with the largest revenue?
select distinct City, sum(Total) as total_revenue from sales group by City order by total_revenue desc ;
-- -- What product line had the largest VAT?
select distinct `Product line`,sum(`Tax 5%`) as high_VAT from sales group by `Product line` order by high_VAT desc ;
-- Fetch each product line and add a column to those product line showing "Good", "Bad". Good if its greater than average sales
alter table sales add column remark varchar(10) ;
update sales set remark = case WHEN (SUM(Quantity) > AVG(Quantity)) THEN 'Good' ELSE 'Bad' END;
select * from sales ;
-- Which branch sold more products than average product sold?
select distinct Branch, sum(Quantity) as qty from sales group by Branch having sum(Quantity) > (select avg(Quantity) from sales) ;
-- What is the most common product line by gender?
select Gender,`Product line`,count(Gender) as cnt from sales group by Gender,`Product line` order by cnt desc ;
-- -- What is the average rating of each product line
select distinct `Product line`, round(avg(Rating),2) as rate from sales group by `Product line` order by rate desc ;
-- -- How many unique customer types does the data have?
select count(distinct `Customer type`) from sales ;
---- How many unique payment methods does the data have?
select count(distinct Payment) from sales ;
-- -- What is the most common customer type?
select distinct `Customer type`, count(`Customer type`) as cnt from sales group by `Customer type` order by cnt desc ;
-- -- Which customer type buys the most?
select `Customer type`,count(*) from sales group by `Customer type` ;
-- -- What is the gender of most of the customers?
select Gender, count(*) as gender_count from sales group by Gender order by gender_count desc ; 
-- -- What is the gender distribution per branch?
select distinct Gender,count(*) as cnt  from sales where Branch = "A" group by Gender order by cnt desc ;
-- -- Which time of the day do customers give most ratings?
select * from sales ;
select time_of_day, count(Rating) as cnt from sales group by time_of_day order by cnt desc ;
select time_of_day, avg(Rating) as avg_rating from sales group by time_of_day order by avg_rating desc ;
-- -- Which time of the day do customers give most ratings per branch?
select time_of_day, count(Rating) as cnt from sales where Branch = "B" group by time_of_day order by cnt desc ;
-- -- Which day of the week has the best avg ratings?
select day_name, round(avg(Rating),2) as average_rating from sales group by day_name order by average_rating desc ;
-- -- Which day of the week has the best average ratings per branch?
select day_name, round(avg(Rating),2) as average_rating from sales where Branch= "B" group by day_name order by average_rating desc ;
-- -- Number of sales made in each time of the day per weekday 
select time_of_day,count(*) as total_count from sales where day_name="Tuesday" group by time_of_day order by total_count desc ; 
-- -- Which of the customer types brings the most revenue?
select `Customer type`,round(sum(Total),2) as total_revenue from sales group by `Customer type` order by total_revenue desc ;
-- -- Which city has the largest tax/VAT percent?
select City, avg(`Tax 5%`) as VAT from sales group by City order by VAT desc ;
-- -- Which customer type pays the most in VAT?
select `Customer type`,round(avg(`Tax 5%`),2) as VAT from sales group by `Customer type` order by VAT desc ;
-- What is the gender distribution per branch?
select Gender, count(*) as cnt from sales where Branch="B" group by Gender order by cnt desc ;
