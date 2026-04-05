type YearDatum = {
  year: number;
};

type CartesianGridGeneratorProps = {
  xAxis?: {
    scale?: ((value: number) => number) | undefined;
  };
};

export type YearAxisGuides = {
  majorYears: number[];
  minorYears: number[];
};

export function buildYearAxisGuides(data: YearDatum[]): YearAxisGuides {
  const years = Array.from(
    new Set(
      data
        .map((row) => row.year)
        .filter((year): year is number => Number.isFinite(year)),
    ),
  ).sort((a, b) => a - b);

  if (years.length === 0) {
    return {
      majorYears: [],
      minorYears: [],
    };
  }

  const minYear = years[0];
  const maxYear = years[years.length - 1];
  const minorYears: number[] = [];
  for (let year = minYear; year <= maxYear; year += 1) {
    minorYears.push(year);
  }

  const majorYears = minorYears.filter((year) => year % 5 === 0);
  return {
    majorYears: majorYears.length > 0 ? majorYears : years,
    minorYears,
  };
}

export function buildVerticalYearCoordinatesGenerator(years: number[]) {
  return ({ xAxis }: CartesianGridGeneratorProps): number[] => {
    const scale = xAxis?.scale;
    if (typeof scale !== "function") {
      return [];
    }
    return years
      .map((year) => scale(year))
      .filter((value): value is number => Number.isFinite(value));
  };
}
