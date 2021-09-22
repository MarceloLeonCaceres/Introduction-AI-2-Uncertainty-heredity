import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data/family0.csv")
    people = load_data(sys.argv[1])
    #people = load_data("data/family0.csv")

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    } 

    # Loop over all sets of people who might have the trait
    names = set(people)
    #print(f"names {names}")
    #print(f"powerset(names) {powerset(names)}")
        
    for have_trait in powerset(names):
        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        
        if fails_evidence:
            continue

        # print("******************************")
        # print(f"fails_evidence: {fails_evidence:}    have_trait: {have_trait}")
        # print()
        #i=1
        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):                
                p = joint_probability(people, one_gene, two_genes, have_trait)
                #print(f"{one_gene} {two_genes} {have_trait} {p}")
                update(probabilities, one_gene, two_genes, have_trait, p)
                #i+=1
    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """    
    p = float(1)

    for person in people:

        if person in two_genes:
            genes = 2
        elif person in one_gene:
            genes = 1
        else:
            genes = 0

        trait = person in have_trait
        madre = people[person]["mother"]
        padre = people[person]["father"]

        # No hay datos de los padres, es la probabilidad incondicional
        if madre is None and padre is None:
            p *= PROBS["gene"][genes]

        # Si hay datos de los padres, 
        else:
            # heredaGen = {madre: 0, padre: 0}
            if madre in two_genes:
                pGenMadre = 1 - PROBS["mutation"]
            elif madre in one_gene:
                pGenMadre = 0.5
            else:
                pGenMadre = PROBS["mutation"]

            if padre in two_genes:
                pGenPadre = 1 - PROBS["mutation"]
            elif padre in one_gene:
                pGenPadre = 0.5
            else:
                pGenPadre = PROBS["mutation"]

            # revisar el ejemplo de https://cs50.harvard.edu/ai/2020/projects/2/heredity/
            if genes == 2:
                p*= (pGenPadre * pGenMadre)
            elif genes == 1:
                p*= ( (pGenMadre*(1-pGenPadre)) + (pGenPadre*(1-pGenMadre)) )
            else:
                p*= ( (1-pGenPadre)*(1-pGenMadre) )

        p *= PROBS["trait"][genes][trait]

    return p
    # raise NotImplementedError


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities.keys():
        if person in one_gene:
            probabilities[person]["gene"][1] += p
        elif person in two_genes:            
            probabilities[person]["gene"][2] += p
        else:   
            probabilities[person]["gene"][0] += p
        
        if person in have_trait:
            probabilities[person]["trait"][True] += p
        else : 
            probabilities[person]["trait"][False] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    personas = probabilities.keys()  
    for person in personas:
        sumProbGenes = probabilities[person]["gene"][0] + probabilities[person]["gene"][1] + probabilities[person]["gene"][2]
        if sumProbGenes > 0:
            probabilities[person]["gene"][0] /= sumProbGenes
            probabilities[person]["gene"][1] /= sumProbGenes
            probabilities[person]["gene"][2] /= sumProbGenes
        
        SumProbTrait = probabilities[person]["trait"][False] + probabilities[person]["trait"][True]
        if SumProbTrait > 0:
            probabilities[person]["trait"][False] /= SumProbTrait
            probabilities[person]["trait"][True]  /= SumProbTrait
    #raise NotImplementedError


if __name__ == "__main__":
    main()
