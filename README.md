### pachyderm
```
helm upgrade -i pachyderm ./apps/pachyderm/dev -n mlops-dev --create-namespace --dry-run
helm upgrade -i pachyderm . -n mlops-dev --create-namespace --dry-run
```

### mlflow
TODO:
- volume NFS expansion
```
```

### changes to make helm work
Changed annotation meta.helm.sh/release-namespace mlops to mlops-dev
- psp
- clusterrole


#### Get real NFS paths in the server
```
alias k="kubectl"
k config set-context --current --namespace=mediamtx-dev # or kubens mediamtx-dev
export labels="mlflow,pachyderm-proxy"
pvs=$(k get pvc $(k get pods -l 'app in (mlflow,pachyderm-proxy)' -o=jsonpath="{range .items[*]}{.spec.volumes[0].persistentVolumeClaim.claimName}{'\n'}{end}") -o jsonpath="{range .items[*]}{.spec.volumeName}{'\n'}{end}")
for pv in $pvs; do
    echo $pv '|' $(k get pv $pv -o jsonpath='{.spec.nfs.path}'| sed 's|^\(/[^/]*\)/\([^/]*\)/\(.*\)$|/data\1/\3|')
done
```
